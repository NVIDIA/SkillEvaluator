# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static, network-free validation of runnable MCP server declarations.

Bundle-reference *provider* MCP entries (``agent_plugin.yaml`` ``mcp``) are
validated by the Pydantic :class:`~skillevaluator.models.plugin.PluginManifest`
model (name charset + provider allowlist). This module adds the blocking Tier 1
security checks for *runnable* MCP servers declared in a contained
``.claude-plugin/plugin.json`` ``mcpServers`` map -- command / url / transport /
env -- plus public-compatible shape checks for contained provider-only entries.

Nothing here launches a process or opens a socket: declarations are inspected
purely as data. Runtime MCP connectivity is a separate Tier 3 concern.

Beyond the blocking shape checks, each declaration is also classified for
supply-chain pinning (:func:`classify_mcp_pinning`), checked for agent-CLI
permission-bypass flags and dangerous environment overrides, and its URL host
is checked against a network-free endpoint policy
(:func:`classify_endpoint_host`). The endpoint policy inspects IP literals and
well-known names only: DNS resolution and HTTP redirects are never evaluated.
"""

from __future__ import annotations

import ipaddress
import itertools
import math
import re
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import parse_qs, unquote, urljoin, urlparse, urlunparse

import idna

from skillevaluator.constants import (
    PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE,
    PLUGIN_CODEX_MANIFEST_TYPE,
    PLUGIN_CURSOR_MANIFEST_TYPE,
)
from skillevaluator.models.plugin import MCP_NAME_PATTERN
from skillevaluator.models.result import Finding, Severity
from skillevaluator.utils.structured_data import MAX_STRUCTURED_NODES

CATEGORY = "MCP_DECLARATION"

# A runnable MCP server speaks one of these transports.
ALLOWED_MCP_TRANSPORTS: frozenset[str] = frozenset({"stdio", "http", "sse"})
# Network MCP endpoints must use a secure scheme; plaintext/dangerous schemes are
# rejected outright.
ALLOWED_MCP_URL_SCHEMES: frozenset[str] = frozenset({"https", "wss"})
# Schemes that can read local files or execute code -- never valid for an MCP URL.
_DANGEROUS_URL_SCHEMES: frozenset[str] = frozenset({"file", "javascript", "data", "gopher", "ftp", "ftps"})
# Plaintext transport schemes -- rejected as insecure (downgrade / MITM surface).
_INSECURE_URL_SCHEMES: frozenset[str] = frozenset({"http", "ws"})

_MCP_NAME_RE = re.compile(MCP_NAME_PATTERN)

# Shell metacharacters that enable command chaining, substitution, or redirection.
# MCP stdio commands are exec'd argv-style (not through a shell), so these have no
# legitimate purpose in a command/arg and indicate injection or shell smuggling.
_SHELL_METACHAR_RE = re.compile(r"[;&|`\n\r]|\$\(|<\(|>\(|&&|\|\||[<>]")
# In a plain argv argument only these still signal shell use: an argument that is just an operator, or
# command or process substitution and line breaks.
_SHELL_OPERATOR_ARG_RE = re.compile(r"^(?:&&|\|\||;;?|\|&?|&|[0-9]?>>?&?[0-9]?|&>>?|<<?<?)$")
_SUBSTITUTION_ARG_RE = re.compile(r"\$\(|`|<\(|>\(|[\n\r]")
# Interpreters invoked with an inline program string execute arbitrary code.
_SHELL_INTERPRETERS: frozenset[str] = frozenset(
    {"sh", "bash", "zsh", "dash", "ksh", "mksh", "ash", "fish", "csh", "tcsh"}
)
# Options of other interpreters that run a program given inline ('python -c CODE', 'node -e CODE'). Such a
# program is code, not an argument, so its text gets the shell-metacharacter check.
_INLINE_CODE_OPTIONS: dict[str, frozenset[str]] = {
    "python": frozenset({"-c"}),
    "node": frozenset({"-e", "--eval", "-p", "--print"}),
    "bun": frozenset({"-e", "--eval", "-p", "--print"}),
    "perl": frozenset({"-e", "-E"}),
    "ruby": frozenset({"-e"}),
    "php": frozenset({"-r"}),
    "lua": frozenset({"-e"}),
    "osascript": frozenset({"-e"}),
}
_INTERPRETER_ALIASES: dict[str, str] = {"nodejs": "node", "py": "python"}
_PYTHON_NAME_RE = re.compile(r"^(?:python|pypy)[0-9.]*$")
# Interpreter options that take a value before the inline program ('python -W ignore -c CODE').
_INTERPRETER_VALUE_OPTIONS = frozenset({"-W", "-X", "-r", "--require", "--import"})
# Moving tags: a dist-tag, branch, or image tag that names a release channel, not one release. They are
# matched only where the package runner reads a version or tag (after the package name's '@', or the image's
# ':tag'), never as a substring, so '@nextui-org/mcp@1.0.0', 'mod.cli:main', or an e-mail are not floating.
_FLOATING_TAGS: frozenset[str] = frozenset(
    {
        "latest",
        "next",
        "canary",
        "beta",
        "alpha",
        "rc",
        "main",
        "master",
        "head",
        "trunk",
        "nightly",
        "edge",
        "dev",
        "develop",
        "unstable",
        "experimental",
        "snapshot",
        "preview",
        "insiders",
        "stable",
    }
)

# Command flags that disable TLS/cert verification.
_INSECURE_TLS_FLAGS: frozenset[str] = frozenset(
    {"--insecure", "-k", "--no-check-certificate", "--tls-no-verify", "--ssl-no-verify", "--no-verify-tls"}
)

# Environment references. Claude Code expands '${NAME}' and '${NAME:-default}' in an MCP server's command,
# args, env, url, and headers (the default when NAME is unset); hooks and many configs also use '$NAME';
# Cursor documents '${env:NAME}'. A reference with no default (or an empty one) carries no value of its own.
_ENV_EXPANSION_RE = re.compile(
    r"\$\{(?P<cursor>env:)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}"
    r"|\$(?P<bare>[A-Za-z_][A-Za-z0-9_]*)"
)
# Only the braced forms are expanded inside an MCP URL.
_URL_EXPANSION_RE = re.compile(r"\$\{(?P<cursor>env:)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")
# What is left of an Authorization-style value once its references are removed: 'Bearer ${TOKEN}'.
_AUTH_SCHEME_ONLY_RE = re.compile(r"(?i)^\s*(?:bearer|basic|token)?\s*$")
# The MCP client that loads a manifest format decides how environment references in its MCP config expand.
McpClient = Literal["claude", "codex", "cursor", "agent_plugins"]
_CLIENT_BY_MANIFEST: dict[str, McpClient] = {
    PLUGIN_CODEX_MANIFEST_TYPE: "codex",
    PLUGIN_CURSOR_MANIFEST_TYPE: "cursor",
    PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE: "agent_plugins",
}
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


# Whole keys (env names, query parameters, flag names) that name a credential on their own: '--auth', a cloud
# URL signature ('X-Amz-Signature'), and the bare nouns.
_CREDENTIAL_KEYS_EXACT = frozenset(
    {
        "auth",
        "x-amz-signature",
        "x-goog-signature",
        "apikey",
        "passwd",
        "password",
        "secret",
        "token",
        "credential",
        "credentials",
    }
)
# Keys that name a credential in a URL query (Google's 'key=', a signed URL's 'sig='), but are often a setting as
# a flag or env name ('--key primary', '--pass 2', '--signature sha256'). There the value must also look like
# secret material.
_QUERY_CREDENTIAL_KEYS = frozenset({"key", "sig", "signature", "pass", "pwd"})
# A last key word that makes a credential noun a setting about the credential, not the credential:
# PASSWORD_POLICY, DB_PASSWORD_FILE, SECRET_NAME, API_KEY_PATH, PASSWORD_MIN_LENGTH.
_NON_CREDENTIAL_QUALIFIERS = frozenset(
    {
        "policy",
        "file",
        "path",
        "dir",
        "directory",
        "name",
        "env",
        "var",
        "variable",
        "type",
        "kind",
        "length",
        "len",
        "min",
        "max",
        "url",
        "uri",
        "endpoint",
        "header",
        "format",
        "mode",
        "rotation",
        "expiry",
        "expires",
        "expiration",
        "ttl",
        "hint",
        "prompt",
        "required",
        "enabled",
        "disabled",
        "strength",
        "algorithm",
        "alg",
        "location",
        "ref",
        "command",
        "cmd",
        "helper",
        "source",
        "store",
        "provider",
        "manager",
        "field",
        "count",
        "age",
        "days",
        "interval",
        "timeout",
        "issuer",
        "audience",
        "scope",
        "scopes",
        "method",
        "version",
    }
)
# Values that are a setting, not secret material, even under a credential-named key: booleans, auth modes,
# and paths ('PASSWORD=true', '--auth basic', 'PRIVATE_KEY=/run/secrets/key').
_NEUTRAL_CREDENTIAL_VALUES = frozenset(
    {
        "true",
        "false",
        "yes",
        "no",
        "on",
        "off",
        "none",
        "null",
        "basic",
        "bearer",
        "digest",
        "oauth",
        "oauth2",
        "oidc",
        "token",
        "apikey",
        "api-key",
        "api_key",
        "ntlm",
        "kerberos",
        "negotiate",
        "anonymous",
        "required",
        "optional",
        "enabled",
        "disabled",
        "jwt",
        "mtls",
    }
)
# Where a path starts: '/', '~/' or '~user/', './' or '../', 'C:\', or '${VAR}/'.
_PATH_VALUE_RE = re.compile(
    r"^(?:/|~[A-Za-z0-9._-]*[/\\]|~$|\.{1,2}[/\\]|[A-Za-z]:[/\\]|\$\{?[A-Za-z_][A-Za-z0-9_]*\}?[/\\])"
)
_PATH_SEPARATOR_RE = re.compile(r"[/\\]+")
_FILE_EXTENSION_RE = re.compile(r"\.[A-Za-z][A-Za-z0-9]{0,7}$")
_BASE64_TEXT_RE = re.compile(r"^[A-Za-z0-9+/=_-]+$")
# Symbols that ordinary names use; any other symbol counts as a character class of its own.
_NAME_SYMBOL_RE = re.compile(r"[^A-Za-z0-9._:/\\-]")
_KEY_WORD_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
# A long random-looking value (hex, or mixed-case base64-like) under a key that names nothing secret.
_RANDOM_VALUE_RE = re.compile(r"^[A-Za-z0-9+/=_-]{32,}$")
_HEX_VALUE_RE = re.compile(r"^[0-9a-fA-F]+$")
_UUID_VALUE_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
# Key words that say a random-looking value is an identifier or a hash, not a credential.
_IDENTIFIER_KEY_WORDS = frozenset(
    {"sha", "hash", "digest", "checksum", "commit", "rev", "revision", "version", "uuid", "guid", "id", "etag"}
)


def _key_words(key: str) -> list[str]:
    """Lower-case words of an env name, header, flag, or camelCase key (``clientSecret`` -> client, secret)."""
    return [word.lower() for word in _KEY_WORD_RE.findall(str(key))]


def _is_credential_key(key: str, *, query: bool = False) -> bool:
    """Whether a key names a credential (``API_KEY``, ``clientSecret``, ``--auth``), not a setting about one.

    With ``query`` (a URL query parameter) the short keys ``key``, ``sig``,
    ``signature``, ``pass``, and ``pwd`` count too.
    """
    text = str(key).strip().lstrip("-")
    if text.lower() in _CREDENTIAL_KEYS_EXACT or (query and text.lower() in _QUERY_CREDENTIAL_KEYS):
        return True
    if not _SECRET_KEY_RE.search(text):
        return False
    words = _key_words(text)
    return not (words and words[-1] in _NON_CREDENTIAL_QUALIFIERS)


def _is_query_credential_key(key: str) -> bool:
    """A short key (``key``, ``sig``, ``pass``) that names a credential in a URL query, but not on its own elsewhere."""
    return str(key).strip().lstrip("-").lower() in _QUERY_CREDENTIAL_KEYS


def _looks_like_secret_material(value: str) -> bool:
    """Whether a value reads as key material by its shape, whatever its key says.

    Three character classes out of lower case, upper case, digits, and other
    symbols (``rawFAKEvalue0123``, ``S3cret!pw``), or a long random hex or
    base64 token. A word, a number, or a name such as ``primary``, ``v2``, or
    ``sha256`` is not.
    """
    text = value.strip()
    if len(text) < 8 or any(char.isspace() for char in text):
        return False
    classes = sum(1 for pattern in (r"[a-z]", r"[A-Z]", r"[0-9]") if re.search(pattern, text))
    classes += 1 if _NAME_SYMBOL_RE.search(text) else 0
    return classes >= 3 or _looks_like_random_secret("value", text)


def _looks_like_path(text: str) -> bool:
    """A file path (``/run/secrets/db``, ``./key.pem``, ``~/.config/x``), not a token that starts with ``/``.

    A base64 secret starts with ``/`` about once in 64, and may hold more
    slashes. One name after the prefix that reads as key material, or a long
    random base64 text, is a token.
    """
    match = _PATH_VALUE_RE.match(text)
    if match is None:
        return False
    rest = text[match.end() :]
    segments = [segment for segment in _PATH_SEPARATOR_RE.split(rest) if segment]
    if len(segments) == 1:
        segment = segments[0]
        return bool(_FILE_EXTENSION_RE.search(segment)) or not _looks_like_secret_material(segment)
    if len(segments) > 1 and _BASE64_TEXT_RE.match(rest) and len(rest) >= 32:
        classes = sum(1 for pattern in (r"[a-z]", r"[A-Z]", r"[0-9]") if re.search(pattern, rest))
        return not (classes == 3 and _shannon_entropy(rest) >= 4.5)
    return True


def _is_neutral_value(value: str) -> bool:
    """A boolean, an auth mode, or a path: never secret material by itself."""
    text = value.strip()
    return text.lower() in _NEUTRAL_CREDENTIAL_VALUES or _looks_like_path(text)


def _shannon_entropy(text: str) -> float:
    counts: dict[str, int] = {}
    for char in text:
        counts[char] = counts.get(char, 0) + 1
    return -sum(count / len(text) * math.log2(count / len(text)) for count in counts.values())


def _looks_like_random_secret(key: str, value: str) -> bool:
    """A neutral-key value that still looks like generated key material (``SERVICE_SEED=9f2c...``)."""
    text = value.strip()
    if not _RANDOM_VALUE_RE.match(text) or _UUID_VALUE_RE.match(text):
        return False
    if _IDENTIFIER_KEY_WORDS & set(_key_words(key)):
        return False
    if _HEX_VALUE_RE.match(text):
        return _shannon_entropy(text) >= 3.0
    classes = sum(1 for pattern in (r"[a-z]", r"[A-Z]", r"[0-9]") if re.search(pattern, text))
    return classes == 3 and _shannon_entropy(text) >= 4.0


def _finding(
    severity: Severity, check_name: str, message: str, file_path: str, suggestion: str, *, name: str | None = None
) -> Finding:
    return Finding(
        category=CATEGORY,
        severity=severity,
        check_name=check_name,
        message=(f"mcpServers['{name}']: {message}" if name else message),
        file_path=file_path,
        suggestion=suggestion,
        # Machine-readable server attribution for the plugin component inventory.
        metadata=({"mcp_server": name} if isinstance(name, str) else {}),
    )


def _is_env_reference(value: str) -> bool:
    """True when *value* is one environment reference that carries no literal value of its own.

    ``$VAR``, ``${VAR}``, ``${VAR:-}`` (an empty default), and Cursor's
    ``${env:VAR}`` qualify. ``${VAR:-default}`` with a non-empty default does not:
    the default ships with the plugin, so it is checked like any literal.
    """
    match = _ENV_EXPANSION_RE.fullmatch(value.strip())
    return match is not None and not match.group("default")


def env_defaults_text(value: str) -> str:
    """The text *value* carries when every referenced variable is unset: defaults stay, references go."""
    return _ENV_EXPANSION_RE.sub(lambda match: match.group("default") or "", value)


def _looks_like_inline_secret(key: str, value: str, *, query: bool = False) -> bool:
    """True when an env/header value is an inline credential rather than a reference.

    With ``query`` the key is a URL query parameter or an HTTP header name, where
    a short key such as ``key`` or ``sig`` names a credential on its own.
    """
    v = value.strip()
    if not v or _is_env_reference(v):
        return False
    if _ENV_EXPANSION_RE.search(v):
        # Judge only what the plugin ships: the literal text around the references and their defaults.
        literal = env_defaults_text(v)
        if _AUTH_SCHEME_ONLY_RE.match(literal):
            return False
        return _looks_like_inline_secret(key, literal, query=query)
    if _SECRET_VALUE_RE.search(v):
        return True
    # An inline HTTP auth-scheme credential ("Bearer <token>" / "Basic <base64>"),
    # independent of the key name -- covers Authorization-style headers.
    if _INLINE_AUTH_SCHEME_RE.match(v):
        return True
    # A credential-named key whose value is a non-empty literal that is not a setting (true, basic, a path).
    if _is_credential_key(str(key), query=query):
        return not _is_neutral_value(v)
    # A short key such as KEY or '--pass' counts only with a value shaped like key material.
    return _is_query_credential_key(str(key)) and not _is_neutral_value(v) and _looks_like_secret_material(v)


def is_env_reference(value: str) -> bool:
    """Public alias: ``True`` when *value* is a pure ``$VAR`` / ``${VAR}`` / ``${VAR:-}`` / ``${env:VAR}`` reference."""
    return _is_env_reference(value)


def mcp_client_for_manifest(manifest_type: str | None) -> McpClient:
    """The client whose MCP config rules apply to a manifest format (Claude Code by default)."""
    return _CLIENT_BY_MANIFEST.get(manifest_type or "", "claude")


def looks_like_inline_secret(key: str, value: str, *, query: bool = True) -> bool:
    """Public alias: ``True`` when a keyed value is an inline credential rather than a reference.

    Callers check URL query parameters and HTTP headers (hooks, Codex
    ``http_headers``), so ``query`` defaults to ``True``: ``?key=...`` names a
    credential. Pass ``False`` for an env name or a command flag.
    """
    return _looks_like_inline_secret(key, value, query=query)


def _credential_flag_name(token: str) -> str | None:
    """Return the flag name when *token* is a credential-bearing option flag.

    Handles ``--api-key`` / ``--api-key=VALUE`` (and short ``-x`` / ``-x=VALUE``)
    forms. The flag name (leading dashes stripped) is matched against the same
    credential vocabulary used for env keys (:func:`_is_credential_key`).
    """
    if not token.startswith("-"):
        return None
    flag = token.lstrip("-").split("=", 1)[0]
    return flag if flag and (_is_credential_key(flag) or _is_query_credential_key(flag)) else None


# Everything after 'scheme:' (and any slashes or backslashes) through the last '@'
# before the path: userinfo to urllib, and to WHATWG clients, which also read
# 'https:user:pw@host' and 'https://user:pw\@host' as carrying it.
_URL_USERINFO_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*:[/\\]*)[^/?#]*@")


# The same userinfo anywhere in a text: inside '${VAR:-https://user:pw@host}', after another word, or after a
# scheme-less '//'. A query or fragment that follows a URL anywhere in a text.
# Inside other text only an authority introduced by two slashes counts, so 'mcp:1.2.3@sha256:...' or
# 'npm:pkg@1.2.3' is left alone.
_EMBEDDED_USERINFO_RE = re.compile(r"(?i)((?:\b[a-z][a-z0-9+.\-]*:)?[/\\]{2,})[^\s/\\?#@{}'\"<>]*@")
_EMBEDDED_URL_QUERY_RE = re.compile(
    r"(?i)(\b[a-z][a-z0-9+.\-]*:[/\\]{2,}(?:\$\{[^}]*\}|[^\s?#'\"<>{}])*)\?(?:\$\{[^}]*\}|[^\s#'\"<>{}])*"
)
_EMBEDDED_URL_FRAGMENT_RE = re.compile(
    r"(?i)(\b[a-z][a-z0-9+.\-]*:[/\\]{2,}(?:\$\{[^}]*\}|[^\s#'\"<>{}])*)#(?:\$\{[^}]*\}|[^\s'\"<>{}])*"
)
# Values that read as credentials wherever they appear in an echoed command or spec: 'Bearer <token>', the
# value after a credential-named flag ('--api-key X', '--token=X'), and a credential-named assignment.
_ECHOED_AUTH_VALUE_RE = re.compile(r"(?i)\b(bearer|basic|token)(\s+)[A-Za-z0-9+/._=~-]{8,}")
_ECHOED_CREDENTIAL_FLAG_RE = re.compile(
    r"(?i)((?<![\w-])--?[\w-]*(?:token|secret|passw(?:or)?d|api[-_]?key|access[-_]?key|private[-_]?key|auth)[\w-]*"
    r"(?:=|\s+))(?!\$)[^\s'\"|;&]+"
)
_ECHOED_CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b[A-Z0-9_]*(?:TOKEN|SECRET|PASSW(?:OR)?D|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY)[A-Z0-9_]*=)(?!\$)[^\s'\"|;&]+"
)
_MAX_USERINFO_PASSES = 4


def _strip_embedded_url_secrets(text: str, *, fragments: bool = True) -> str:
    """``text`` with the userinfo and query (and, with ``fragments``, the fragment) of every URL in it removed."""
    for _pass in range(_MAX_USERINFO_PASSES):
        stripped = _EMBEDDED_USERINFO_RE.sub(r"\1", text)
        if stripped == text:
            break
        text = stripped
    text = _EMBEDDED_URL_QUERY_RE.sub(r"\1", text)
    return _EMBEDDED_URL_FRAGMENT_RE.sub(r"\1", text) if fragments else text


def redact_secrets(text: str) -> str:
    """``text`` (a command token, spec, or line) with credentials removed, for finding messages.

    URL userinfo and queries are dropped (a fragment such as a git ref stays), and
    known secret shapes (``ghp_...``, ``sk-...``, a JWT, ``Bearer <token>``) become
    ``<redacted>``.
    """
    text = _strip_embedded_url_secrets(text, fragments=False)
    text = _SECRET_VALUE_RE.sub("<redacted>", text)
    text = _ECHOED_AUTH_VALUE_RE.sub(r"\1\2<redacted>", text)
    text = _ECHOED_CREDENTIAL_FLAG_RE.sub(r"\1<redacted>", text)
    return _ECHOED_CREDENTIAL_ASSIGNMENT_RE.sub(r"\1<redacted>", text)


def redacted_url(url: str) -> str:
    """URL for finding messages: scheme, host, port, and path only.

    Userinfo, parameters, query, and fragment are dropped so an inline credential
    is never echoed into reports or CI logs, however the authority is written,
    and also when the URL sits inside an environment reference
    (``${MCP_URL:-https://user:pw@host/mcp}``) or after other text.
    """
    text = url.strip()
    if _URL_EXPANSION_RE.search(text):
        return _redacted_reference_url(text)
    try:
        parsed = urlparse(_URL_USERINFO_RE.sub(r"\1", text, count=1))
    except ValueError:  # e.g. an unbalanced '[' in the authority
        return "<unparseable URL>"
    shown = urlunparse((parsed.scheme, parsed.netloc.rpartition("@")[2], parsed.path, "", "", ""))
    return _strip_embedded_url_secrets(shown)


def _url_display_mask(url: str) -> list[bool]:
    """For each character of ``url``: whether a finding may show it (scheme, host, port, and path).

    User information (through the last ``@`` of the authority) and everything
    from the first ``?`` or ``#`` on are hidden.
    """
    keep = [True] * len(url)
    scheme = _URL_SCHEME_RE.match(url)
    start = scheme.end() if scheme else 0
    authority = start
    while authority < len(url) and url[authority] in "/\\":
        authority += 1
    if scheme or authority > start:
        ends = [index for index in (url.find(mark, authority) for mark in "/?#") if index != -1]
        at = url.rfind("@", authority, min(ends, default=len(url)))
        for index in range(authority, at + 1):
            keep[index] = False
    cut = min((index for index in (url.find("?"), url.find("#")) if index != -1), default=len(url))
    for index in range(cut, len(url)):
        keep[index] = False
    return keep


def _redacted_reference_url(text: str) -> str:
    """:func:`redacted_url` for a URL that holds ``${NAME}`` or ``${NAME:-default}`` references.

    What is hidden is decided on the URL the client builds, with every default
    in place, so a default in the path (``${P:-/mcp?token=...}``), a query after
    a reference (``${BASE:-https://host}/mcp?token=...``), or user information
    split over references is never shown. The references stay readable:
    ``${MCP_URL:-https://host/mcp}``.
    """
    pieces: list[tuple[str, re.Match[str] | None]] = []
    position = 0
    for match in _URL_EXPANSION_RE.finditer(text):
        if match.start() > position:
            pieces.append((text[position : match.start()], None))
        default = match.group("default")
        # A reference with no default (or Cursor's form, never expanded here) stands for some host-like text.
        pieces.append((default if default and not match.group("cursor") else _ENV_MARKER, match))
        position = match.end()
    if position < len(text):
        pieces.append((text[position:], None))
    keep = _url_display_mask("".join(piece for piece, _match in pieces))
    shown: list[str] = []
    offset = 0
    for piece, match in pieces:
        kept = "".join(char for char, wanted in zip(piece, keep[offset : offset + len(piece)], strict=True) if wanted)
        offset += len(piece)
        if match is None:
            shown.append(kept)
        elif not kept:
            continue  # the whole reference sits in the user information, query, or fragment
        elif match.group("default") and not match.group("cursor"):
            shown.append(f"${{{match.group('name')}:-{kept}}}")
        elif match.group("default"):
            shown.append(f"${{env:{match.group('name')}:-{redacted_url(match.group('default'))}}}")
        else:
            shown.append(match.group(0))
    return _strip_embedded_url_secrets("".join(shown))


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


def _authority_end(text: str, scheme: str | None) -> int:
    """Index where the path, query, or fragment starts after ``scheme:`` and the authority of a special URL."""
    start = len(scheme) + 1 if scheme else 0
    while start < len(text) and text[start] in "/\\":
        start += 1
    ends = [index for index in (text.find(mark, start) for mark in "/\\?#") if index != -1]
    return min(ends, default=len(text))


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
    # A plain space after the host ('https://host/a b') reads the same way to both parsers: clients
    # percent-encode it. Whitespace before the path, other (Unicode) whitespace, and control characters
    # anywhere can change where the URL points or hide part of it.
    inner_scheme = _url_scheme(inner)
    path_start = _authority_end(inner, inner_scheme) if inner_scheme in _WHATWG_SPECIAL_SCHEMES else len(inner)
    if any(
        unicodedata.category(char) == "Cc" or (char.isspace() and (index < path_start or char != " "))
        for index, char in enumerate(inner)
    ):
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


def _client_reading(parsed: Any) -> str:
    """How WHATWG clients read an ambiguous URL, for messages: no userinfo, query, or fragment."""
    try:
        host = parsed.hostname
        port = parsed.port
    except ValueError:  # e.g. 'https://user:password\@host' is host 'user' with port 'password'
        return "reject it as invalid"
    if not host:
        return "find no host in it"
    display_host = f"[{host}]" if ":" in host else host
    shown = f"{parsed.scheme}://{display_host}{f':{port}' if port else ''}{parsed.path}"
    return f"read it as {shown!r}"


def _safe_hostname(parsed: Any) -> str | None:
    try:
        return parsed.hostname
    except ValueError:
        return None


def _check_url_inline_secrets(
    name: str,
    url: str,
    parsed: Any,
    file_path: str,
    findings: list[Finding],
    *,
    label: str = "url",
    password_only: bool = False,
) -> None:
    """Flag inline credentials embedded in a URL's userinfo or query string (``url`` is only displayed, redacted).

    With ``password_only``, a user name alone does not count (a reading in which
    other text lands in the user name).
    """
    try:
        username, password = parsed.username, parsed.password
    except ValueError:  # malformed netloc / port
        username = password = None
    if password_only:
        username = None
    if (password and not _is_env_reference(password)) or (username and not _is_env_reference(username)):
        findings.append(
            _finding(
                Severity.CRITICAL,
                "mcp_url_inline_secret",
                f"{label} embeds inline userinfo credentials: {redacted_url(url)!r} (userinfo withheld); only "
                "${ENV} references are allowed",
                file_path,
                'Remove user:password@ from the URL; pass credentials by reference (e.g. header "${MY_TOKEN}").',
                name=name,
            )
        )
    for key, values in parse_qs(parsed.query, keep_blank_values=True).items():
        if not _is_credential_key(key, query=True):
            continue
        if any(v and not _is_env_reference(v) and not _is_neutral_value(v) for v in values):
            findings.append(
                _finding(
                    Severity.CRITICAL,
                    "mcp_url_inline_secret",
                    f"{label} query parameter {key!r} carries an inline credential; only ${{ENV}} references are "
                    "allowed",
                    file_path,
                    "Do not put credentials in the URL query string; reference a secret handle/env var instead.",
                    name=name,
                )
            )


def _is_insecure_tls_env(key: str, value: str) -> bool:
    """Detect env pairs that disable TLS/certificate verification."""
    k = str(key).strip().upper()
    v = str(value).strip().lower()
    if k == "NODE_TLS_REJECT_UNAUTHORIZED":
        return v == "0"
    if k == "PYTHONHTTPSVERIFY":
        # CPython disables HTTPS verification ONLY when this is exactly "0"; "" (or
        # absent) and any other value keep verification ON -- flagging those is a FP.
        return v == "0"
    if k in {"GIT_SSL_NO_VERIFY", "CURL_INSECURE", "SSL_NO_VERIFY", "TLS_INSECURE", "SSL_VERIFY_NONE"}:
        return v in {"1", "true", "yes", "on"}
    return False


def _iter_command_tokens(config: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    command = config.get("command")
    if isinstance(command, str):
        tokens.append(command)
    args = config.get("args")
    if isinstance(args, list):
        tokens.extend(str(a) for a in args)
    return tokens


_COMMAND_LINE_PARSERS: dict[str, frozenset[str]] = {
    "cmd": frozenset({"/c", "/k", "/r"}),
    "powershell": frozenset({"-c", "-command", "-commandwithargs"}),
    "pwsh": frozenset({"-c", "-command", "-commandwithargs"}),
}


def _parses_shell_operators(argv: list[str]) -> bool:
    """Whether the arguments of ``argv`` reach a command-line parser (``cmd /c``, ``powershell -Command``)."""
    if not argv:
        return False
    inner = unwrap_launch_command(_launch_argv(argv[0], argv[1:]), expand_shell=False, stop_at=_COMMAND_LINE_PARSERS)
    switches = _COMMAND_LINE_PARSERS.get(_command_basename(inner[0])) if inner else None
    return switches is not None and any(token.lower() in switches for token in inner[1:])


def _interpreter_inline_code(argv: list[str]) -> str | None:
    """The program of ``python -c CODE``, ``node -e CODE``, ``perl -e CODE``, ``deno eval CODE``, and the like."""
    if not argv:
        return None
    base = _command_basename(argv[0])
    base = "python" if _PYTHON_NAME_RE.match(base) else _INTERPRETER_ALIASES.get(base, base)
    if base == "deno":
        return argv[2] if argv[1:2] == ["eval"] and len(argv) > 2 else None
    options = _INLINE_CODE_OPTIONS.get(base)
    if options is None:
        return None
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == "--" or not token.startswith("-") or len(token) == 1:
            return None  # a script or module runs; what follows are its arguments
        name, sep, attached = token.partition("=")
        # The option itself ('-c', '--eval=CODE'), or a short-option cluster that ends in it ('python -Bc CODE').
        if name in options or (not token.startswith("--") and f"-{token[-1]}" in options):
            if sep and name in options:
                return attached
            return argv[index + 1] if index + 1 < len(argv) else None
        index += 2 if token in _INTERPRETER_VALUE_OPTIONS else 1
    return None


def _validate_command(name: str, config: dict[str, Any], file_path: str, findings: list[Finding]) -> None:
    command = config.get("command")
    if not isinstance(command, str) or not command.strip():
        findings.append(
            _finding(
                Severity.HIGH,
                "mcp_command_empty",
                "runnable MCP 'command' must be a non-empty string",
                file_path,
                "Set 'command' to the server executable (argv-style, no shell string).",
                name=name,
            )
        )
        return

    args = config.get("args")
    if args is not None and not isinstance(args, list):
        findings.append(
            _finding(
                Severity.HIGH,
                "mcp_args_not_list",
                "runnable MCP 'args' must be a list of strings",
                file_path,
                "Express command arguments as a JSON array of strings.",
                name=name,
            )
        )

    tokens = _iter_command_tokens(config)
    # The program the launch really runs, with wrappers looked through ('env sh -c', 'timeout 30 bash -lc',
    # 'sudo sh -c', 'cmd /c sh -c'). A shell's '-c' program and an interpreter's inline program ('python -c',
    # 'node -e') are code, so their text is checked like a shell line.
    launched = unwrap_launch_command(_launch_argv(command, tokens[1:]), expand_shell=False)
    runs_shell_program, shell_text_program = (
        _shell_inline(launched) if launched and _command_basename(launched[0]) in _SHELL_INTERPRETERS else (False, None)
    )
    programs = [program for program in (shell_text_program, _interpreter_inline_code(launched)) if program]
    # MCP clients start the server argv-style, without a shell, so an operator inside one argument ('foo|bar',
    # a PEP 440 range '>=1.2,<2') is plain text. Operators count in 'command' itself (a shell line smuggled into
    # the executable), in the arguments of a command-line wrapper that does parse them ('cmd /c', PowerShell),
    # in an inline program, and as an argument that is only an operator ('&&') or carries command substitution
    # ('$(...)', a backtick).
    shell_args = _parses_shell_operators(tokens)
    for index, token in enumerate(tokens):
        shell_text = index == 0 or shell_args or (token in programs)
        if (shell_text and _SHELL_METACHAR_RE.search(token)) or (
            not shell_text and (_SHELL_OPERATOR_ARG_RE.match(token.strip()) or _SUBSTITUTION_ARG_RE.search(token))
        ):
            findings.append(
                _finding(
                    Severity.CRITICAL,
                    "mcp_command_shell_metacharacters",
                    f"command token contains shell metacharacters: {redact_secrets(token)!r}",
                    file_path,
                    "Remove shell operators (; | & ` $() < >) from 'command' and from 'cmd /c' or PowerShell "
                    "command lines; other arguments are passed argv-style, not through a shell.",
                    name=name,
                )
            )
        if token in _INSECURE_TLS_FLAGS:
            findings.append(
                _finding(
                    Severity.CRITICAL,
                    "mcp_command_disables_tls",
                    f"command disables TLS/certificate verification: {token!r}",
                    file_path,
                    "Remove insecure-TLS flags; do not disable certificate verification.",
                    name=name,
                )
            )
    for program in programs:
        # An inline program split out of one command-line string ('env -S "sh -c ..."') is not a token of its own.
        if program not in tokens and _SHELL_METACHAR_RE.search(program):
            findings.append(
                _finding(
                    Severity.CRITICAL,
                    "mcp_command_shell_metacharacters",
                    f"inline program contains shell metacharacters: {redact_secrets(program)!r}",
                    file_path,
                    "Invoke the server binary directly instead of passing it an inline program.",
                    name=name,
                )
            )
    # Inline credentials carried in command arguments. A credential-named flag
    # (--api-key, --token, --password, ...) must reference an env var, never a raw
    # literal; and any argument whose *value* has a known secret shape or is an
    # inline "Bearer/Basic <token>" is flagged regardless of the flag name.
    # ${ENV} references are always allowed.
    arg_list = [str(a) for a in args] if isinstance(args, list) else []
    flagged_value_idx = -1
    for idx, token in enumerate(arg_list):
        flag = _credential_flag_name(token)
        if flag is not None:
            if "=" in token:
                value, value_idx = token.split("=", 1)[1], idx
            elif idx + 1 < len(arg_list) and not arg_list[idx + 1].startswith("-"):
                # A following token that looks like another flag is NOT this flag's
                # value (avoids flagging e.g. `--api-key --verbose`).
                value, value_idx = arg_list[idx + 1], idx + 1
            else:
                value, value_idx = "", -1
            # A setting ('--auth basic', a path) is not a credential; a short flag such as '--key' or '--pass'
            # needs a value shaped like key material ('--key primary' is a setting).
            if value and _looks_like_inline_secret(flag, value):
                findings.append(
                    _finding(
                        Severity.CRITICAL,
                        "mcp_command_inline_secret",
                        f"command argument {flag!r} carries an inline credential; only ${{ENV}} references are allowed",
                        file_path,
                        'Pass the secret by reference (e.g. "${MY_TOKEN}"); never inline a raw credential in args.',
                        name=name,
                    )
                )
                flagged_value_idx = value_idx
            continue
        if idx == flagged_value_idx:
            continue  # already reported as the preceding flag's value
        stripped = token.strip()
        if (
            stripped
            and not _is_env_reference(stripped)
            and (_SECRET_VALUE_RE.search(stripped) or _INLINE_AUTH_SCHEME_RE.match(stripped))
        ):
            findings.append(
                _finding(
                    Severity.CRITICAL,
                    "mcp_command_inline_secret",
                    f"command argument args[{idx}] contains an inline credential (value not shown); only ${{ENV}} "
                    "references are allowed",
                    file_path,
                    'Pass the secret by reference (e.g. "${MY_TOKEN}"); never inline a raw credential in args.',
                    name=name,
                )
            )

    # Shell interpreter invoked with an inline program string ('sh -c "..."', 'bash -lc', 'env sh -c', 'sudo sh -c').
    if runs_shell_program:
        shell, wrapper = launched[0], _launch_argv(command, tokens[1:])[0]
        through = "" if wrapper == shell else f" through {redact_secrets(wrapper)!r}"
        findings.append(
            _finding(
                Severity.CRITICAL,
                "mcp_command_dangerous_form",
                f"command invokes a shell interpreter with '-c' ({redact_secrets(shell)!r}{through}); this executes "
                "an arbitrary program string",
                file_path,
                "Invoke the server binary directly instead of wrapping it in a shell '-c' string.",
                name=name,
            )
        )


# Stand-in for a reference with no default, so the URL can be parsed to see what the reference decides.
_ENV_MARKER = "zzenvrefzz"


def _expand_url_defaults(url: str) -> tuple[str, list[str]]:
    """An MCP URL as Claude Code expands it when no variable is set.

    ``${NAME:-default}`` becomes its default. ``${NAME}``, ``${NAME:-}``, and
    Cursor's ``${env:NAME}`` stay; their names are returned, because their value
    comes from the user's environment.
    """
    unresolved: list[str] = []

    def _expand(match: re.Match[str]) -> str:
        default = match.group("default")
        if default and not match.group("cursor"):
            return default
        unresolved.append(match.group("name"))
        return match.group(0)

    return _URL_EXPANSION_RE.sub(_expand, url), unresolved


def expand_url_defaults(url: str) -> str:
    """The URL with every ``${NAME:-default}`` replaced by its default; references without a default are kept."""
    return _expand_url_defaults(url)[0]


def _env_controls_endpoint(expanded: str) -> bool:
    """Whether a reference without a default decides the scheme, host, or port of ``expanded``."""
    probe = _URL_EXPANSION_RE.sub(_ENV_MARKER, expanded)
    try:
        parsed = urlparse(whatwg_url(probe))
        host = parsed.hostname or ""
        port = str(parsed.port or "")
    except ValueError:
        return True
    return not parsed.scheme or any(_ENV_MARKER in part for part in (parsed.scheme.lower(), host, port))


def _url_secrets_in_text(name: str, url: str, text: str, label: str, file_path: str, findings: list[Finding]) -> None:
    """Credentials the URL text ships (also in a default), when the URL is not otherwise checked."""
    try:
        parsed = urlparse(text.strip())
    except ValueError:
        return
    _check_url_inline_secrets(name, url, parsed, file_path, findings, label=label)


def _resolve_url_references(
    name: str, url: str, label: str, file_path: str, findings: list[Finding], client: McpClient
) -> str | None:
    """The URL to check for an MCP URL that uses environment expansion, or ``None`` when it cannot be checked.

    Claude Code (and, leniently, Cursor and Agent Plugins clients) expand
    ``${NAME:-default}``: the default the plugin ships is checked like any URL.
    A reference without a default that decides the scheme or host leaves the
    endpoint to the user, which is a LOW advisory. Codex does not expand
    references in plugin MCP URLs at all (it reads the text as a relative URL
    and never connects), which is HIGH.
    """
    expanded, unresolved = _expand_url_defaults(url)
    shown = redacted_url(url)
    if client == "codex":
        effect = (
            "reads it as a relative URL and never connects to the server"
            if _env_controls_endpoint(_URL_EXPANSION_RE.sub("${UNSET}", url))
            else "sends the '${...}' text literally, so the variable is never filled in"
        )
        findings.append(
            _finding(
                Severity.HIGH,
                "mcp_url_env_not_expanded",
                f"{label} {shown!r} uses '${{...}}' environment expansion, which Codex does not apply to plugin "
                f"MCP URLs; Codex {effect}",
                file_path,
                "Write the full https:// endpoint in a Codex plugin; '${VAR:-default}' expansion is a Claude Code "
                "feature.",
                name=name,
            )
        )
        _url_secrets_in_text(name, url, expanded, label, file_path, findings)
        return None
    if unresolved and _env_controls_endpoint(expanded):
        variables = ", ".join(dict.fromkeys(unresolved))
        findings.append(
            _finding(
                Severity.LOW,
                "mcp_url_env_unchecked",
                f"{label} {shown!r} takes its scheme or host from environment variable(s) {variables} with no "
                "default, so the endpoint the client reaches cannot be checked statically",
                file_path,
                'Give the variable a safe default ("${VAR:-https://host/mcp}") or document the value users must set.',
                name=name,
            )
        )
        _url_secrets_in_text(name, url, expanded, label, file_path, findings)
        return None
    return expanded


def _is_loopback_host(host: str | None) -> bool:
    endpoint = classify_endpoint_host(host) if host else None
    return endpoint is not None and endpoint.reason == "loopback"


def _validate_oauth_urls(
    name: str,
    config: dict[str, Any],
    file_path: str,
    findings: list[Finding],
    allowed_private_hosts: Iterable[str],
    *,
    client: McpClient,
) -> None:
    """Claude Code fetches OAuth server metadata from ``oauth.authServerMetadataUrl``: the same URL policy applies."""
    oauth = config.get("oauth")
    if not isinstance(oauth, dict):
        return
    url = oauth.get("authServerMetadataUrl")
    if isinstance(url, str) and url.strip():
        _check_endpoint_url(
            name, url, "oauth.authServerMetadataUrl", file_path, findings, allowed_private_hosts, client=client
        )


def _validate_url(
    name: str,
    config: dict[str, Any],
    file_path: str,
    findings: list[Finding],
    allowed_private_hosts: Iterable[str] = (),
    *,
    client: McpClient = "claude",
) -> None:
    url = config.get("url")
    if not isinstance(url, str) or not url.strip():
        findings.append(
            _finding(
                Severity.HIGH,
                "mcp_url_empty",
                "runnable MCP 'url' must be a non-empty string",
                file_path,
                "Set 'url' to the server endpoint using a secure https:// (or wss://) URL.",
                name=name,
            )
        )
        return
    _check_endpoint_url(name, url, "url", file_path, findings, allowed_private_hosts, client=client)


def _check_endpoint_url(
    name: str,
    url: str,
    label: str,
    file_path: str,
    findings: list[Finding],
    allowed_private_hosts: Iterable[str],
    *,
    client: McpClient = "claude",
) -> None:
    """Scheme, host, credential, and endpoint checks for one URL field of an MCP server (``label`` names it)."""
    display = url  # messages show the declared text, redacted
    if _URL_EXPANSION_RE.search(url):
        expanded = _resolve_url_references(name, url, label, file_path, findings, client)
        if expanded is None:
            return
        url = expanded
    shown = redacted_url(display)  # messages never echo userinfo or query credentials
    try:
        # ``raw`` is how urllib (and Python clients) read the text; ``parsed`` is how
        # WHATWG clients (Node and Rust MCP clients) read it. They differ only for an
        # ambiguous URL, which is flagged below.
        raw = urlparse(url.strip())
        parsed = urlparse(whatwg_url(url))
    except ValueError:  # e.g. an unbalanced '[' in the authority
        findings.append(
            _finding(
                Severity.HIGH,
                "mcp_url_malformed_authority",
                f"{label} could not be parsed (malformed authority)",
                file_path,
                "Use a valid host[:port] authority, e.g. https://host:443/path.",
                name=name,
            )
        )
        return
    problems = url_ambiguities(url)
    if problems:
        findings.append(
            _finding(
                Severity.HIGH,
                "mcp_url_malformed_authority",
                f"{label} contains {', and '.join(problems)}, so MCP clients and URL parsers disagree on where it "
                f"points: {shown!r} (WHATWG clients {_client_reading(parsed)})",
                file_path,
                "Write the URL with '//' after the scheme and without backslashes, whitespace, or control "
                "characters, e.g. https://host/path.",
                name=name,
            )
        )
    scheme = (parsed.scheme or "").lower()
    # Host findings name the URL the way the client that connects reads it: for
    # 'https://169.254.169.254\\@pub.example/' that is the metadata host, not pub.example.
    client_shown = redacted_url(whatwg_url(url)) if problems else shown
    # Inline credentials in userinfo/query are persisted verbatim; check them
    # independent of the scheme (secure https URLs are the common case). The
    # userinfo is the one WHATWG clients send (also in 'https:user:password@host').
    found = len(findings)
    _check_url_inline_secrets(name, display, parsed, file_path, findings, label=label)
    if problems and len(findings) == found:
        # urllib reads another authority. A literal password there still ships in the plugin text, but text
        # before a backslash that urllib reads as a user name ('169.254.169.254\\@host') is not a credential.
        _check_url_inline_secrets(name, display, raw, file_path, findings, label=label, password_only=True)
    if problems:
        # A Python client may still connect where urllib reads the host: classify that one too.
        try:
            raw_host = raw.hostname
        except ValueError:
            raw_host = None
        if raw_host and raw_host != _safe_hostname(parsed):
            _validate_endpoint(
                name,
                display,
                raw_host,
                file_path,
                findings,
                allowed_private_hosts,
                label=label,
                shown=f"{shown} (as Python clients read it)",
            )
    if scheme in ALLOWED_MCP_URL_SCHEMES:
        # A secure scheme alone is not a usable endpoint: require a host to connect
        # to, and reject a malformed authority/port. Otherwise a URL like "https://"
        # passes Tier 1 and only fails later in Harbor. Both are static, no network.
        try:
            host = parsed.hostname
            _ = parsed.port  # property access raises ValueError on a malformed port
        except ValueError:
            findings.append(
                _finding(
                    Severity.HIGH,
                    "mcp_url_malformed_authority",
                    f"{label} has a malformed authority/port: {shown!r}",
                    file_path,
                    "Use a valid host[:port] authority, e.g. https://host:443/path.",
                    name=name,
                )
            )
            return
        if not host:
            findings.append(
                _finding(
                    Severity.HIGH,
                    "mcp_url_no_host",
                    f"{label} uses scheme {scheme!r} but has no host to connect to: {shown!r}",
                    file_path,
                    "Provide a full endpoint with a hostname, e.g. https://host[:port]/path.",
                    name=name,
                )
            )
        _validate_endpoint(
            name, display, host, file_path, findings, allowed_private_hosts, label=label, shown=client_shown
        )
        return
    if scheme in _INSECURE_URL_SCHEMES:
        # Plaintext endpoints are blocked below; still report where they point.
        try:
            insecure_host = parsed.hostname
        except ValueError:
            insecure_host = None
        _validate_endpoint(
            name, display, insecure_host, file_path, findings, allowed_private_hosts, label=label, shown=client_shown
        )
    if scheme == "":
        findings.append(
            _finding(
                Severity.HIGH,
                "mcp_url_scheme_missing",
                f"{label} {shown!r} has no scheme, so MCP clients cannot connect to it",
                file_path,
                "Write the full endpoint with its scheme, e.g. https://host/mcp.",
                name=name,
            )
        )
    elif scheme in _DANGEROUS_URL_SCHEMES:
        findings.append(
            _finding(
                Severity.CRITICAL,
                "mcp_url_dangerous_scheme",
                f"{label} uses a dangerous scheme {scheme!r}: {shown!r}",
                file_path,
                "Use a secure https:// or wss:// endpoint; file/data/javascript/ftp schemes are not permitted.",
                name=name,
            )
        )
    elif scheme in _INSECURE_URL_SCHEMES and _is_loopback_host(_safe_hostname(parsed)):
        # Same rule as HTTP hooks: plaintext to this machine's loopback interface does not cross a network
        # (a local desktop app's MCP server); the MEDIUM loopback finding above still reports it.
        pass
    elif scheme in _INSECURE_URL_SCHEMES:
        findings.append(
            _finding(
                Severity.HIGH,
                "mcp_url_insecure_scheme",
                f"{label} uses an insecure plaintext scheme {scheme!r}: {shown!r}",
                file_path,
                "Use https:// (or wss://) so the MCP transport is encrypted.",
                name=name,
            )
        )
    else:
        findings.append(
            _finding(
                Severity.HIGH,
                "mcp_url_scheme_not_allowed",
                f"{label} scheme {scheme!r} is not an allowed MCP scheme: {shown!r}",
                file_path,
                f"Use one of the allowed secure schemes: {', '.join(sorted(ALLOWED_MCP_URL_SCHEMES))}.",
                name=name,
            )
        )


# Codex MCP fields that carry auth or environment the evaluation runtime does not apply (Tier 3 then reports
# the server INCOMPLETE); kept in step with plugin_components._CODEX_UNAPPLIED_MCP_FIELDS.
_CODEX_UNAPPLIED_FIELDS = ("env_vars", "env_http_headers", "bearer_token_env_var", "http_headers_helper", "oauth")


def _validate_env_and_headers(
    name: str, config: dict[str, Any], file_path: str, findings: list[Finding], *, client: McpClient = "claude"
) -> None:
    for section in ("env", "headers"):
        block = config.get(section)
        if block is None:
            continue
        if not isinstance(block, dict):
            findings.append(
                _finding(
                    Severity.HIGH,
                    "mcp_env_not_object",
                    f"'{section}' must be an object mapping names to reference values",
                    file_path,
                    f"Express '{section}' as a JSON object of key -> value.",
                    name=name,
                )
            )
            continue
        # NON-BLOCKING advisory: the wrapper runtime applies command+args (stdio) and
        # url (http/sse) only -- Harbor's per-MCP-server config has no env/headers field.
        # Only a native Claude Code arm (the plugin's own .mcp.json) applies them. The
        # inline-secret / insecure-TLS checks below still run, so a raw credential
        # declared here is still caught and blocks.
        findings.append(
            _finding(
                Severity.LOW,
                "mcp_field_ignored",
                f"'{section}' is applied only when Tier 3 loads the plugin natively in Claude Code; the wrapper "
                "runtime and the other harnesses ignore it, and such a Tier 3 run of this server is reported "
                "INCOMPLETE",
                file_path,
                f"Use --plugin-load native with Claude Code, or remove '{section}' and rely on task-level "
                "environment / CI credential injection.",
                name=name,
            )
        )
        _validate_secret_values(name, section, block, file_path, findings, tls_env=True)
    oauth = config.get("oauth")
    if isinstance(oauth, dict):
        # A literal client secret in the OAuth block ships with the plugin like any other credential.
        _validate_secret_values(name, "oauth", oauth, file_path, findings, tls_env=False)
    if client == "codex":
        for field in _CODEX_UNAPPLIED_FIELDS:
            if config.get(field):
                findings.append(
                    _finding(
                        Severity.LOW,
                        "mcp_field_ignored",
                        f"Codex '{field}' is not applied by the evaluation runtime; a Tier 3 run of this server is "
                        "reported INCOMPLETE",
                        file_path,
                        f"Expect an INCOMPLETE Tier 3 run, or remove '{field}' if the server works without it.",
                        name=name,
                    )
                )


def _validate_secret_values(
    name: str, section: str, block: dict[Any, Any], file_path: str, findings: list[Finding], *, tls_env: bool
) -> None:
    """Inline credentials (and, for env and headers, TLS-off switches) among one block's string values."""
    for key, value in block.items():
        if not isinstance(value, str):
            continue
        if tls_env and _is_insecure_tls_env(key, value):
            findings.append(
                _finding(
                    Severity.CRITICAL,
                    "mcp_insecure_tls_env",
                    f"'{section}.{key}' disables TLS/certificate verification",
                    file_path,
                    "Do not disable TLS verification via environment variables.",
                    name=name,
                )
            )
        # A header name such as 'Key' names a credential the way a query key does; an env name 'KEY' needs a
        # value shaped like key material.
        if _looks_like_inline_secret(key, value, query=section == "headers"):
            findings.append(
                _finding(
                    Severity.CRITICAL,
                    "mcp_inline_secret",
                    f"'{section}.{key}' contains an inline credential; only ${{ENV}} references are allowed",
                    file_path,
                    'Reference a secret handle/env var (e.g. "${MY_TOKEN}"); never inline a raw secret.',
                    name=name,
                )
            )
        elif _looks_like_random_secret(key, env_defaults_text(value)):
            findings.append(
                _finding(
                    Severity.MEDIUM,
                    "mcp_possible_inline_secret",
                    f"'{section}.{key}' holds a long random-looking value that may be a credential (value not "
                    "shown); the key name does not say what it is",
                    file_path,
                    'If it is a secret, reference it (e.g. "${MY_SECRET}"); if not, rename the key to say what it '
                    "holds.",
                    name=name,
                )
            )


def _validate_transport(name: str, config: dict[str, Any], file_path: str, findings: list[Finding]) -> None:
    raw = config.get("transport", config.get("type"))
    if raw is None:
        return
    if not isinstance(raw, str) or raw.strip().lower() not in ALLOWED_MCP_TRANSPORTS:
        findings.append(
            _finding(
                Severity.HIGH,
                "mcp_transport_invalid",
                f"transport {raw!r} is not one of {sorted(ALLOWED_MCP_TRANSPORTS)}",
                file_path,
                f"Set transport to one of: {', '.join(sorted(ALLOWED_MCP_TRANSPORTS))}.",
                name=name,
            )
        )
        return

    literal = raw.strip()
    canonical = literal.lower()
    # Harbor's transport literal is case-sensitive: the agent adapter compares it
    # against the exact lowercase "stdio"/"http"/"sse" and the persist path writes
    # it verbatim, so a value Tier 1 accepts must be the exact form Harbor accepts.
    if literal != canonical:
        findings.append(
            _finding(
                Severity.HIGH,
                "mcp_transport_bad_casing",
                f"transport {raw!r} must be lowercase {canonical!r}; Harbor's transport literal is case-sensitive",
                file_path,
                f"Use the exact lowercase transport literal {canonical!r}.",
                name=name,
            )
        )

    # Kind <-> transport consistency: a stdio server is launched from a 'command';
    # an http/sse server is reached over a 'url'. Harbor rejects a transport that
    # contradicts the declared kind (http/sse need a url; stdio needs a command).
    has_command = "command" in config
    has_url = "url" in config
    if has_command and not has_url and canonical != "stdio":
        findings.append(
            _finding(
                Severity.HIGH,
                "mcp_transport_kind_mismatch",
                f"command (stdio) server declares transport {raw!r}; a command server must use transport 'stdio'",
                file_path,
                "Set transport to 'stdio' (or omit it) for command-based MCP servers.",
                name=name,
            )
        )
    elif has_url and not has_command and canonical not in {"http", "sse"}:
        findings.append(
            _finding(
                Severity.HIGH,
                "mcp_transport_kind_mismatch",
                f"url server declares transport {raw!r}; a url server must use transport 'http' or 'sse'",
                file_path,
                "Set transport to 'http' or 'sse' for url-based MCP servers.",
                name=name,
            )
        )


def _validate_insecure_tls_config(name: str, config: dict[str, Any], file_path: str, findings: list[Finding]) -> None:
    """Reject config keys that turn off TLS/certificate verification."""
    if config.get("insecure") is True:
        findings.append(
            _finding(
                Severity.CRITICAL,
                "mcp_insecure_flag",
                "'insecure: true' disables endpoint security",
                file_path,
                "Remove 'insecure'; connect over a verified TLS endpoint.",
                name=name,
            )
        )
    for section in ("tls", "ssl"):
        block = config.get(section)
        if not isinstance(block, dict):
            continue
        if block.get("rejectUnauthorized") is False or block.get("verify") is False:
            findings.append(
                _finding(
                    Severity.CRITICAL,
                    "mcp_insecure_tls_config",
                    f"'{section}' disables certificate verification (rejectUnauthorized/verify = false)",
                    file_path,
                    "Do not disable certificate verification; use a valid certificate chain.",
                    name=name,
                )
            )


# --------------------------------------------------------------------------- #
# Supply-chain pinning                                                        #
# --------------------------------------------------------------------------- #
PinStatus = Literal["pinned", "unpinned", "not_applicable"]

_EXACT_SEMVER_RE = re.compile(r"^v?\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")
_PEP440_EXACT_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?\s*(?:===\s*\S+|==\s*[0-9][0-9A-Za-z.!+_-]*)$"
)
_GIT_SHA_RE = re.compile(r"(?:#|@)[0-9a-fA-F]{40}(?:$|[&#])")
_DOCKER_DIGEST_RE = re.compile(r"@sha256:[0-9a-fA-F]{64}$")
_VERSION_TAG_RE = re.compile(r"^v?\d+(?:\.\d+)*(?:[-+._][0-9A-Za-z.]+)*$")
_DOTTED_VERSION_PREFIX_RE = re.compile(r"^v?\d+\.\d+(?![0-9A-Za-z])")
_PLUGIN_PATH_REFS: tuple[str, ...] = ("${CLAUDE_PLUGIN_ROOT}", "${CLAUDE_PLUGIN_DATA}", "${CLAUDE_PROJECT_DIR}")
_LOCAL_SPEC_PREFIXES: tuple[str, ...] = (".", "/", "~", "file:", *_PLUGIN_PATH_REFS)
_REMOTE_SPEC_PREFIXES: tuple[str, ...] = ("git+", "git:", "github:", "gitlab:", "bitbucket:", "http://", "https://")

# Value-taking flags per package runner, so a flag's value is never mistaken for
# the package spec. Unknown flags are treated as boolean.
_NPX_VALUE_FLAGS = frozenset(
    {
        "-p",
        "--package",
        "-c",
        "--call",
        "--registry",
        "--cache",
        "--userconfig",
        "--globalconfig",
        "--prefix",
        "-w",
        "--workspace",
        "--loglevel",
        "--node-options",
        "--script-shell",
        "--include",
        "--omit",
        "--before",
    }
)
_DLX_VALUE_FLAGS = frozenset(
    {"-p", "--package", "--registry", "--allow-build", "-C", "--dir", "--reporter", "--loglevel", "--filter"}
)
# Options that take a value before the 'run'/'exec'/'dlx' subcommand of a runner front end.
_NPM_GLOBAL_VALUE_FLAGS = frozenset({*_NPX_VALUE_FLAGS, "-C", "--dir"})
_UV_RUN_VALUE_FLAGS = frozenset(
    {
        "--with",
        "--with-editable",
        "--with-requirements",
        "-p",
        "--python",
        "--project",
        "--directory",
        "--package",
        "--extra",
        "--group",
        "--only-group",
        "--no-group",
        "--env-file",
        "--index",
        "--default-index",
        "-i",
        "--index-url",
        "--extra-index-url",
        "-f",
        "--find-links",
        "--config-file",
        "--cache-dir",
    }
)
_GO_RUN_VALUE_FLAGS = frozenset({"-C", "-exec", "-o", "-p", "-tags", "-ldflags", "-gcflags", "-mod", "-modfile"})
_DNX_VALUE_FLAGS = frozenset({"--version", "--source", "--add-source", "--configfile", "--verbosity", "-v"})
_UVX_VALUE_FLAGS = frozenset(
    {
        "--from",
        "--with",
        "--with-editable",
        "--with-requirements",
        "-p",
        "--python",
        "--index",
        "--default-index",
        "-i",
        "--index-url",
        "--extra-index-url",
        "-f",
        "--find-links",
        "-c",
        "--constraints",
        "--overrides",
        "--build-constraints",
        "--env-file",
        "--directory",
        "--project",
        "--config-file",
        "--cache-dir",
        "--color",
        "--python-preference",
        "--keyring-provider",
        "--index-strategy",
        "--resolution",
        "--prerelease",
        "--exclude-newer",
        "--link-mode",
        "-P",
        "--upgrade-package",
        "--reinstall-package",
        "--refresh-package",
    }
)
_PIPX_VALUE_FLAGS = frozenset({"--spec", "--python", "--pip-args", "--index-url", "--fetch-python"})
_DENO_VALUE_FLAGS = frozenset({"-c", "--config", "--import-map", "--lock", "--cert", "--location", "--seed"})
_DOCKER_VALUE_FLAGS = frozenset(
    {
        "-a",
        "--attach",
        "--add-host",
        "--annotation",
        "--blkio-weight",
        "--blkio-weight-device",
        "--cap-add",
        "--cap-drop",
        "--cgroup-parent",
        "--cgroupns",
        "--cidfile",
        "--cpu-period",
        "--cpu-quota",
        "--cpu-rt-period",
        "--cpu-rt-runtime",
        "-c",
        "--cpu-shares",
        "--cpus",
        "--cpuset-cpus",
        "--cpuset-mems",
        "--detach-keys",
        "--device",
        "--device-cgroup-rule",
        "--device-read-bps",
        "--device-read-iops",
        "--device-write-bps",
        "--device-write-iops",
        "--dns",
        "--dns-option",
        "--dns-search",
        "--domainname",
        "--entrypoint",
        "-e",
        "--env",
        "--env-file",
        "--expose",
        "--gpus",
        "--group-add",
        "--health-cmd",
        "--health-interval",
        "--health-retries",
        "--health-start-interval",
        "--health-start-period",
        "--health-timeout",
        "-h",
        "--hostname",
        "--ip",
        "--ip6",
        "--ipc",
        "--isolation",
        "--kernel-memory",
        "-l",
        "--label",
        "--label-file",
        "--link",
        "--link-local-ip",
        "--log-driver",
        "--log-opt",
        "--mac-address",
        "-m",
        "--memory",
        "--memory-reservation",
        "--memory-swap",
        "--memory-swappiness",
        "--mount",
        "--name",
        "--net",
        "--network",
        "--network-alias",
        "--oom-score-adj",
        "--pid",
        "--pids-limit",
        "--platform",
        "-p",
        "--publish",
        "--pull",
        "--restart",
        "--runtime",
        "--security-opt",
        "--shm-size",
        "--stop-signal",
        "--stop-timeout",
        "--storage-opt",
        "--sysctl",
        "--tmpfs",
        "--ulimit",
        "-u",
        "--user",
        "--userns",
        "--uts",
        "-v",
        "--volume",
        "--volume-driver",
        "--volumes-from",
        "-w",
        "--workdir",
    }
)
_CONTAINER_RUNTIMES = frozenset({"docker", "podman", "nerdctl"})
_CONTAINER_GLOBAL_VALUE_FLAGS = frozenset({"-c", "--context", "-H", "--host", "--config", "-l", "--log-level"})
# Programs that a launch command wraps around the real one: their own options (and, for timeout, the
# duration) come first, then the wrapped command line.
_WRAPPER_VALUE_FLAGS: dict[str, frozenset[str]] = {
    "env": frozenset({"-u", "--unset", "-C", "--chdir"}),
    "timeout": frozenset({"-s", "--signal", "-k", "--kill-after"}),
    "nice": frozenset({"-n", "--adjustment"}),
    "nohup": frozenset(),
    "sudo": frozenset({"-u", "--user", "-g", "--group", "-C", "-D", "--chdir", "-h", "--host", "-p", "--prompt"}),
    "doas": frozenset({"-u", "-C"}),
    "exec": frozenset({"-a"}),
    "command": frozenset(),
    "time": frozenset({"-f", "--format", "-o", "--output"}),
    "stdbuf": frozenset({"-i", "-o", "-e"}),
    "busybox": frozenset(),
}
# Shell options whose value is the program (fish), and long options that take a value.
_SHELL_COMMAND_OPTIONS = frozenset({"--command"})
_SHELL_LONG_VALUE_OPTIONS = frozenset({"--rcfile", "--init-file", "--init-command"})
_MAX_UNWRAP_DEPTH = 8
_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# Everything the pinning classifier knows by name; a 'command' with spaces that ends in one of
# these is a path ('C:\\Program Files\\nodejs\\npx.cmd'), not a whole command line.
_KNOWN_LAUNCHERS = frozenset(
    {
        "npx",
        "bunx",
        "pnpx",
        "bun",
        "pnpm",
        "yarn",
        "npm",
        "uvx",
        "uv",
        "pipx",
        "deno",
        "go",
        "dnx",
        "cmd",
        "powershell",
        "pwsh",
        *_CONTAINER_RUNTIMES,
        *_SHELL_INTERPRETERS,
        *_WRAPPER_VALUE_FLAGS,
    }
)


@dataclass(frozen=True)
class McpPinning:
    """Static supply-chain pinning classification of one MCP declaration.

    ``floating`` marks an ``unpinned`` launch whose version or tag names a
    moving release channel (``@latest``, ``:main``, ``:1-nightly``); it is the
    blocking case. Other ``unpinned`` launches (no version, a range) warn.
    """

    status: PinStatus
    detail: str
    floating: bool = False

    @property
    def pinned(self) -> bool | None:
        """``True``/``False`` for package runners; ``None`` when not applicable."""
        if self.status == "not_applicable":
            return None
        return self.status == "pinned"


def _command_basename(command: str) -> str:
    base = command.strip().strip("\"'").replace("\\", "/").rsplit("/", 1)[-1].lower()
    for suffix in (".exe", ".cmd", ".bat", ".ps1"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _is_floating_tag(tag: str, *, composite: bool = True) -> bool:
    """Whether a version or tag names a moving channel: ``latest``, ``main``, and, with ``composite``, ``1-latest``."""
    lowered = tag.strip().lower()
    if lowered in _FLOATING_TAGS:
        return True
    return composite and any(part in _FLOATING_TAGS for part in re.split(r"[-+._]", lowered) if part)


def _is_floating_image_tag(tag: str) -> bool:
    """Whether an image tag names a moving channel: ``latest``, ``1-latest``, ``0-nightly``, ``stable-alpine``.

    A tag that starts with a version of at least ``major.minor`` names one
    release, also with a pre-release word (``1.0.0-beta``, ``2.1.0-rc``,
    ``3.0-dev``, ``2024.01-preview``), as ``@1.0.0-beta`` does for npm. Only
    ``latest`` moves wherever it appears (``1.2-latest``).
    """
    lowered = tag.strip().lower()
    if not _is_floating_tag(lowered):
        return False
    if _DOTTED_VERSION_PREFIX_RE.match(lowered):
        return "latest" in re.split(r"[-+._]", lowered)
    return True


def _split_command_line(tokens: list[str]) -> list[str]:
    """A command line given as one string (``cmd /c "npx -y pkg"``) as words; other argv as is."""
    if len(tokens) == 1 and any(char.isspace() for char in tokens[0]):
        return [word.strip("\"'") for word in tokens[0].split()]
    return list(tokens)


def _launch_argv(command: str, args: list[str]) -> list[str]:
    """The argv a launch config runs: ``command`` plus ``args``.

    A ``command`` with spaces is a whole command line (``"npx -y pkg"``) unless it
    is a path to a known launcher (``C:\\Program Files\\nodejs\\npx.cmd``).
    """
    stripped = command.strip()
    if any(char.isspace() for char in stripped) and _command_basename(stripped) not in _KNOWN_LAUNCHERS:
        return [*_split_command_line([stripped]), *args]
    return [stripped, *args]


def _skip_options(tokens: list[str], value_flags: frozenset[str]) -> int:
    """Index of the first positional token, skipping options and the values of ``value_flags``."""
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            return index + 1
        if not token.startswith("-") or len(token) == 1:
            return index
        if "=" not in token and token in value_flags:
            index += 1
        index += 1
    return index


def _shell_inline(tokens: list[str]) -> tuple[bool, str | None]:
    """``(runs a -c program, the program)`` for a shell argv.

    Reads ``sh -c '<program>'``, combined flags (``bash -lc``, ``sh -ec``),
    the ``+c`` form bash, sh, and zsh also accept, options before or after
    ``-c`` (``bash -o pipefail -c``, ``bash -oe pipefail -c``, ``bash -c -e``),
    ``--`` before the program, and fish's ``--command``. Options end at the
    first operand (a script's own ``-config`` is not the shell's). The program
    is the first word after the options; it is ``None`` when none follows.
    """
    inline = False
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            index += 1
            break
        if token in _SHELL_COMMAND_OPTIONS:
            return True, (tokens[index + 1] if index + 1 < len(tokens) else None)
        if token.startswith(tuple(f"{option}=" for option in _SHELL_COMMAND_OPTIONS)):
            return True, token.partition("=")[2]
        if token.startswith("--"):
            index += 2 if token in _SHELL_LONG_VALUE_OPTIONS else 1
            continue
        if len(token) > 1 and token[0] in "-+":
            letters = token[1:]
            # bash, sh, and zsh also run '+c' (and '+lc') as '-c'.
            inline = inline or "c" in letters
            # '-o pipefail', '-euo pipefail', '-oe pipefail', '+O extglob': the option name is the next word.
            index += 2 if "o" in letters or "O" in letters else 1
            continue
        break
    if not inline:
        return False, None
    return True, (tokens[index] if index < len(tokens) else None)


def _shell_program(tokens: list[str]) -> str | None:
    """The program string of ``sh -c '<program>'`` (also ``bash -lc``); ``None`` otherwise."""
    return _shell_inline(tokens)[1]


def shell_program(tokens: list[str]) -> str | None:
    """The program string when ``tokens`` run a shell interpreter with ``-c`` (``bash -c '<program>'``)."""
    if not tokens or _command_basename(tokens[0]) not in _SHELL_INTERPRETERS:
        return None
    return _shell_program([str(token) for token in tokens])


def unwrap_launch_command(tokens: list[str], *, expand_shell: bool = True, stop_at: Iterable[str] = ()) -> list[str]:
    """Strip launch wrappers so the runner they start is classified.

    ``cmd /c <command line>``, ``powershell -Command <line>``, ``env [opts]
    NAME=VALUE ...``, ``timeout <duration>``, ``nice``, ``nohup``, ``sudo``,
    ``doas``, ``exec``, ``command``, ``time``, and ``stdbuf`` are removed, along
    with their own options. With ``expand_shell``, ``sh -c '<program>'`` is
    replaced by the program's words (its first command only; hook analysis
    splits shell programs into commands itself and passes ``False``).
    Unwrapping stops at a launcher named in ``stop_at``.
    """
    current = [str(token) for token in tokens]
    stops = frozenset(stop_at)
    for _depth in range(_MAX_UNWRAP_DEPTH):
        if not current:
            return current
        base = _command_basename(current[0])
        rest = current[1:]
        if base in stops:
            return current
        if base == "cmd":
            switch = next((index for index, token in enumerate(rest) if token.lower() in {"/c", "/k", "/r"}), None)
            if switch is None:
                return current
            current = _split_command_line(rest[switch + 1 :])
        elif base in {"powershell", "pwsh"}:
            switch = next(
                (index for index, token in enumerate(rest) if token.lower() in {"-c", "-command", "-commandwithargs"}),
                None,
            )
            if switch is None:
                return current
            current = _split_command_line([" ".join(rest[switch + 1 :])]) if rest[switch + 1 :] else []
        elif base in _SHELL_INTERPRETERS and expand_shell:
            program = _shell_program(current)
            if program is None:
                return current
            current = _split_command_line([program]) if program.strip() else []
        elif base == "env":
            split_string = _flag_values(rest, ("-S", "--split-string"))
            if split_string:
                current = _split_command_line([split_string[0]])
                continue
            index = 0
            while index < len(rest) and (rest[index].startswith("-") or _ENV_ASSIGNMENT_RE.match(rest[index])):
                if rest[index] in _WRAPPER_VALUE_FLAGS["env"]:
                    index += 1
                index += 1
            current = rest[index:]
        elif base in _WRAPPER_VALUE_FLAGS:
            index = _skip_options(rest, _WRAPPER_VALUE_FLAGS[base])
            if base == "timeout" and index < len(rest):
                index += 1  # the duration
            current = rest[index:]
        else:
            return current
    return current


def _positionals(tokens: list[str], value_flags: frozenset[str]) -> Iterator[tuple[int, str]]:
    """Yield ``(index, token)`` for positional arguments, skipping flags and their values."""
    index = 0
    after_separator = False
    while index < len(tokens):
        token = tokens[index]
        if after_separator:
            yield index, token
        elif token == "--":
            after_separator = True
        elif token.startswith("-") and len(token) > 1:
            if "=" not in token and token in value_flags:
                index += 1  # skip the flag's value
        else:
            yield index, token
        index += 1


def _flag_values(tokens: list[str], names: Iterable[str]) -> list[str]:
    """Return every value passed to one of ``names`` (``--flag v`` or ``--flag=v``)."""
    wanted = frozenset(names)
    values: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            break
        flag, sep, inline = token.partition("=")
        if flag in wanted:
            if sep:
                values.append(inline)
            elif index + 1 < len(tokens):
                values.append(tokens[index + 1])
                index += 1
        index += 1
    return values


def _first_positional(tokens: list[str], value_flags: frozenset[str]) -> str | None:
    return next((token for _index, token in _positionals(tokens, value_flags)), None)


def _is_local_spec(spec: str) -> bool:
    return spec.startswith(_LOCAL_SPEC_PREFIXES)


def _git_ref(spec: str) -> str | None:
    """The branch or tag a git/URL spec names, or ``None`` when it names none.

    npm reads the ref after ``#`` (``github:o/r#main``, ``git+https://h/o/r.git#main``);
    pip and uv read it after the last ``@`` of the URL path
    (``git+https://h/o/r@main``, ``pkg @ git+https://h/o/r@main#egg=pkg``). A user
    name before the host (``git+ssh://git@h/o/r``), a ``key=value`` fragment
    (``#egg=``, ``#subdirectory=``), and npm's ``#semver:`` / ``#path:`` parts are
    not refs.
    """
    start = re.search(r"git\+|git:|github:|gitlab:|bitbucket:|https?://", spec)
    body, _hash, fragment = spec[start.start() if start else 0 :].partition("#")
    for part in re.split(r"&|::", fragment):
        if part and "=" not in part and not part.startswith(("semver:", "path:")):
            return part
    if "://" in body:
        path = body.split("://", 1)[1].partition("/")[2]
    else:
        path = body.split(":", 1)[1] if ":" in body else body
    _repo, at, ref = path.rpartition("@")
    return ref if at and ref else None


def _classify_npm_spec(spec: str) -> McpPinning:
    """Classify an npm package spec (``pkg``, ``@scope/pkg@1.2.3``, git/URL, local path)."""
    if _is_local_spec(spec):
        return McpPinning("not_applicable", f"local package path {spec!r}")
    if spec.startswith("$"):
        return McpPinning("unpinned", f"package spec {spec!r} is taken from an environment reference")
    if spec.startswith(_REMOTE_SPEC_PREFIXES) or (not spec.startswith("@") and "/" in spec):
        if _GIT_SHA_RE.search(spec):
            return McpPinning("pinned", f"git/URL spec pinned to a commit: {spec!r}")
        ref = _git_ref(spec)
        if ref and _is_floating_tag(ref, composite=False):
            return McpPinning(
                "unpinned", f"git/URL spec {spec!r} follows the moving branch or tag {ref!r}", floating=True
            )
        return McpPinning("unpinned", f"git/URL/GitHub spec without a commit SHA: {spec!r}")
    at = spec.find("@", 1) if spec.startswith("@") else spec.find("@")
    if at <= 0:
        return McpPinning("unpinned", f"package {spec!r} has no version (resolves to the latest release)")
    version = spec[at + 1 :]
    if _EXACT_SEMVER_RE.match(version):
        return McpPinning("pinned", f"exact version {spec!r}")
    if "$" in version:
        return McpPinning("unpinned", f"package {spec!r} takes its version from an environment reference")
    # A dist-tag starts with a letter; a range starts with a digit or an operator ('^1.0.0-beta' is a range).
    if _is_floating_tag(version, composite=version[:1].isalpha()):
        return McpPinning("unpinned", f"package {spec!r} uses the floating dist-tag {version!r}", floating=True)
    return McpPinning("unpinned", f"package {spec!r} uses a version range or dist-tag, not an exact version")


def _classify_python_spec(spec: str) -> McpPinning:
    """Classify a PyPI requirement spec as used by ``uvx`` / ``pipx run``."""
    if _is_local_spec(spec):
        return McpPinning("not_applicable", f"local package path {spec!r}")
    if spec.startswith("$"):
        return McpPinning("unpinned", f"package spec {spec!r} is taken from an environment reference")
    if spec.startswith(_REMOTE_SPEC_PREFIXES) or "@ git+" in spec or "@git+" in spec:
        if _GIT_SHA_RE.search(spec) or "#sha256=" in spec:
            return McpPinning("pinned", f"git/URL spec pinned to a commit or hash: {spec!r}")
        ref = _git_ref(spec)
        if ref and _is_floating_tag(ref, composite=False):
            return McpPinning(
                "unpinned", f"git/URL spec {spec!r} follows the moving branch or tag {ref!r}", floating=True
            )
        return McpPinning("unpinned", f"git/URL spec without a commit SHA or hash: {spec!r}")
    if _PEP440_EXACT_RE.match(spec):
        return McpPinning("pinned", f"exact version {spec!r}")
    name, sep, version = spec.partition("@")
    if sep and name and _EXACT_SEMVER_RE.match(version.strip()):
        return McpPinning("pinned", f"exact version {spec!r}")
    if sep and name and _is_floating_tag(version, composite=False):
        # uv reads 'pkg@latest' as "the newest release, refreshed on every run".
        return McpPinning("unpinned", f"package {spec!r} uses the floating version {version.strip()!r}", floating=True)
    if any(marker in spec for marker in ("<", ">", "~", "!", "*", ",", "=", "@")):
        return McpPinning("unpinned", f"requirement {spec!r} is a range or tag, not an exact '==' version")
    return McpPinning("unpinned", f"package {spec!r} has no version (resolves to the latest release)")


_IMAGE_ENV_REF_RE = re.compile(r"\$\{[^}]*\}|\$[A-Za-z_][A-Za-z0-9_]*")


def _classify_image(image: str) -> McpPinning:
    """Classify a container image reference (``repo[:tag][@digest]``)."""
    if _DOCKER_DIGEST_RE.search(image):
        # A digest names one image; a tag next to it ('mcp:latest@sha256:...') is ignored by the runtime.
        return McpPinning("pinned", f"image pinned by digest {image!r}")
    if "$" in image:
        # The name comes from the environment, but a tag written after it ('${IMAGE}:main') still floats.
        literal = _IMAGE_ENV_REF_RE.sub("\x00", image)
        literal_last = literal.rsplit("/", 1)[-1].partition("@")[0]
        literal_tag = literal_last.split(":", 1)[1] if ":" in literal_last else None
        if literal_tag and "\x00" not in literal_tag and _is_floating_image_tag(literal_tag):
            return McpPinning("unpinned", f"image {image!r} uses the floating tag {literal_tag!r}", floating=True)
        return McpPinning("unpinned", f"image {image!r} is taken from an environment reference")
    last, at, digest = image.rsplit("/", 1)[-1].partition("@")
    if at:
        return McpPinning("unpinned", f"image {image!r} has an incomplete digest {digest!r} (not a sha256 digest)")
    tag = last.split(":", 1)[1] if ":" in last else None
    if tag is None:
        return McpPinning("unpinned", f"image {image!r} has no tag or digest (implicit ':latest')")
    if _is_floating_image_tag(tag):
        return McpPinning("unpinned", f"image {image!r} uses the floating tag {tag!r}", floating=True)
    if _VERSION_TAG_RE.match(tag):
        return McpPinning("pinned", f"image {image!r} uses a version tag (tags are mutable; a digest is stronger)")
    return McpPinning("unpinned", f"image {image!r} uses the non-version tag {tag!r}")


def _classify_spec_list(specs: list[str], classify: Any) -> McpPinning:
    """The worst classification of several specs: floating, then unpinned, then pinned."""
    results = [classify(spec) for spec in specs]
    for wanted in (
        lambda result: result.floating,
        lambda result: result.status == "unpinned",
        lambda result: result.status == "pinned",
    ):
        found = next((result for result in results if wanted(result)), None)
        if found is not None:
            return found
    return results[0]


def _subcommand(tokens: list[str], value_flags: frozenset[str]) -> tuple[str | None, list[str]]:
    """``(subcommand, tokens after it)`` of a runner front end, skipping its global options."""
    index = _skip_options(tokens, value_flags)
    if index >= len(tokens):
        return None, []
    return tokens[index], tokens[index + 1 :]


def classify_mcp_pinning(config: Any, *, client: McpClient = "claude") -> McpPinning:
    """Classify whether one MCP declaration runs an exactly-pinned package (see :func:`_classify_launch`).

    Claude Code expands ``${NAME:-default}`` in ``command`` and ``args`` (Cursor
    and Agent Plugins servers are read the same way, and so is a shell running a
    hook command), so the default the plugin ships is what gets classified:
    ``npx -y ${PKG:-@scope/mcp@latest}`` is floating. Codex runs the text as is.
    The detail never carries a credential from the spec it quotes.
    """
    launch = config if client == "codex" else _with_launch_defaults(config)
    pin = _classify_launch(launch)
    if launch is not config:
        pin = McpPinning(pin.status, f"{pin.detail}, with each ${{NAME:-default}} read at its default", pin.floating)
    return _redacted_pin(pin)


def _with_launch_defaults(config: Any) -> Any:
    """``config`` with every ``${NAME:-default}`` in ``command`` and ``args`` replaced by its default.

    The same object comes back when nothing has a default.
    """
    if not isinstance(config, dict) or not isinstance(config.get("command"), str):
        return config
    raw_args = config.get("args")
    fields = [config["command"], *(raw_args if isinstance(raw_args, list) else [])]
    if not any(isinstance(field, str) and _URL_EXPANSION_RE.search(field) for field in fields):
        return config
    expanded = [expand_url_defaults(field) if isinstance(field, str) else field for field in fields]
    if expanded == fields:
        return config
    launch = dict(config, command=expanded[0])
    if isinstance(raw_args, list):
        launch["args"] = expanded[1:]
    return launch


def _classify_launch(config: Any) -> McpPinning:
    """Classify whether one MCP declaration runs an exactly-pinned package.

    Package runners (``npx``, ``bunx``, ``bun x``, ``pnpm dlx``, ``yarn dlx``,
    ``npm exec``, ``uvx``, ``uv tool run``, ``uv run --with``, ``pipx run``,
    ``deno run`` of a registry spec, ``go run pkg@version``, ``dnx``, and
    ``docker|podman|nerdctl run``) are ``pinned`` only with an exact version
    (``pkg@1.2.3``, ``pkg==1.2.3``, ``--from pkg==1.2.3``, ``image:1.2.3``,
    ``image@sha256:...``); otherwise they are ``unpinned`` (``floating`` for a
    moving tag such as ``@latest``). Launch wrappers (``cmd /c``, ``env X=1``,
    ``timeout``, ``sudo``, ``sh -c``) are looked through. Local interpreters and
    scripts (``node ./server.js``, ``python -m local_module``, ``./bin/server``),
    URL servers, and provider-only entries are ``not_applicable``.
    """
    if not isinstance(config, dict):
        return McpPinning("not_applicable", "declaration is not an object")
    command = config.get("command")
    if not isinstance(command, str) or not command.strip():
        if isinstance(config.get("url"), str):
            return McpPinning("not_applicable", "remote url server (no package is installed)")
        return McpPinning("not_applicable", "provider-only or non-runnable declaration")
    raw_args = config.get("args")
    args = [str(arg) for arg in raw_args] if isinstance(raw_args, list) else []
    argv = unwrap_launch_command(_launch_argv(command, args))
    if not argv:
        return McpPinning("not_applicable", "launch wrapper without a command")
    base, args = _command_basename(argv[0]), argv[1:]

    if base in {"npx", "bunx", "pnpx"}:
        return _classify_npm_runner(base, args, _NPX_VALUE_FLAGS if base == "npx" else _DLX_VALUE_FLAGS)
    if base == "bun":
        sub, rest = _subcommand(args, _DLX_VALUE_FLAGS)
        if sub == "x":
            return _classify_npm_runner("bun x", rest, _DLX_VALUE_FLAGS)
    if base in {"pnpm", "yarn"}:
        sub, rest = _subcommand(args, _DLX_VALUE_FLAGS)
        if sub == "dlx":
            # '--package' may come before 'dlx' ('pnpm --package=@scope/pkg dlx bin').
            before = args[: len(args) - len(rest) - 1]
            packages = _flag_values(before, ("-p", "--package"))
            rest = [*(f"--package={spec}" for spec in packages), *rest]
            return _classify_npm_runner(f"{base} dlx", rest, _DLX_VALUE_FLAGS)
    if base == "npm":
        sub, rest = _subcommand(args, _NPM_GLOBAL_VALUE_FLAGS)
        if sub in {"exec", "x"}:
            return _classify_npm_runner("npm exec", rest, _NPX_VALUE_FLAGS)
    if base == "uvx":
        return _classify_uv_tool("uvx", args)
    if base == "uv":
        sub, rest = _subcommand(args, _UV_RUN_VALUE_FLAGS)
        if sub == "tool":
            tool_sub, tool_rest = _subcommand(rest, _UVX_VALUE_FLAGS)
            if tool_sub in {"run", "x"}:
                return _classify_uv_tool("uv tool run", tool_rest)
            return McpPinning("not_applicable", "uv tool invocation is not 'run'")
        if sub == "run":
            # 'uv run' runs from the project environment; '--with' adds packages resolved on every run.
            options = [*args[: len(args) - len(rest) - 1], *rest[: _skip_options(rest, _UV_RUN_VALUE_FLAGS)]]
            with_specs = _flag_values(options, ("--with",))
            if with_specs:
                return _prefixed("uv run --with: ", _classify_spec_list(with_specs, _classify_python_spec))
            return McpPinning("not_applicable", "uv run of the project environment (no '--with' package)")
    if base == "pipx" and args[:1] == ["run"]:
        rest = args[1:]
        spec_values = _flag_values(rest, ("--spec",))
        spec = spec_values[0] if spec_values else _first_positional(rest, _PIPX_VALUE_FLAGS)
        if spec is None:
            return McpPinning("unpinned", "pipx run invocation without a package spec")
        return _prefixed("pipx run: ", _classify_python_spec(spec))
    if base == "deno" and args[:1] == ["run"]:
        spec = _first_positional(args[1:], _DENO_VALUE_FLAGS)
        if spec and spec.startswith(("npm:", "jsr:")):
            return _prefixed("deno run: ", _classify_npm_spec(spec.split(":", 1)[1]))
        if spec and spec.startswith(("http://", "https://")):
            if re.search(r"@v?\d+\.\d+\.\d+(?:[/?#]|$)", spec):
                return McpPinning("pinned", f"deno run: remote module pinned to an exact version: {spec!r}")
            return McpPinning("unpinned", f"deno run: remote module without an exact version: {spec!r}")
        return McpPinning("not_applicable", "deno run of a local script")
    if base == "go" and args[:1] == ["run"]:
        return _classify_go_run(_first_positional(args[1:], _GO_RUN_VALUE_FLAGS))
    if base == "dnx":
        return _classify_dnx(args)
    if base in _CONTAINER_RUNTIMES:
        image = _container_run_image(args)
        if image is False:
            return McpPinning("not_applicable", f"{base} invocation is not 'run'")
        if image is None:
            return McpPinning("unpinned", f"{base} run invocation without an image")
        return _prefixed(f"{base} run: ", _classify_image(str(image)))
    return McpPinning("not_applicable", f"local interpreter, script, or binary ({base!r})")


def _classify_uv_tool(runner: str, rest: list[str]) -> McpPinning:
    """``uvx [opts] pkg`` / ``uv tool run``: the tool spec (or ``--from``) and every ``--with`` requirement."""
    from_values = _flag_values(rest, ("--from",))
    spec = from_values[0] if from_values else _first_positional(rest, _UVX_VALUE_FLAGS)
    if spec is None:
        return McpPinning("unpinned", f"{runner} invocation without a package spec")
    tool = _classify_python_spec(spec)
    # '--with' installs more packages next to the tool, each resolved on every run unless exact. The worst one
    # decides: a floating spec (tool or extra) beats an unpinned one, which beats the tool's own result.
    extras = [_classify_python_spec(extra) for extra in _flag_values(rest, ("--with",))]
    for wanted in (lambda result: result.floating, lambda result: result.status == "unpinned"):
        if wanted(tool):
            return _prefixed(f"{runner}: ", tool)
        worst = next((extra for extra in extras if wanted(extra)), None)
        if worst is not None:
            return _prefixed(f"{runner} --with: ", worst)
    return _prefixed(f"{runner}: ", tool)


def _classify_go_run(spec: str | None) -> McpPinning:
    """``go run module/path@version``: pinned for an exact ``vX.Y.Z``; a local package is not applicable."""
    if spec is None or "@" not in spec:
        return McpPinning("not_applicable", "go run of a local package (versions come from go.mod)")
    version = spec.rpartition("@")[2]
    if re.match(r"^v\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$", version):
        return McpPinning("pinned", f"go run: exact module version {spec!r}")
    if _is_floating_tag(version, composite=False):
        return McpPinning("unpinned", f"go run: module {spec!r} uses the floating query {version!r}", floating=True)
    return McpPinning("unpinned", f"go run: module {spec!r} is not pinned to an exact vX.Y.Z version")


def _classify_dnx(args: list[str]) -> McpPinning:
    """``dnx Package[@version]`` or ``dnx Package --version X`` (.NET tool runner)."""
    spec = _first_positional(args, _DNX_VALUE_FLAGS)
    if spec is None:
        return McpPinning("unpinned", "dnx invocation without a package")
    versions = _flag_values(args, ("--version",))
    version = versions[0] if versions else (spec.rpartition("@")[2] if "@" in spec else "")
    if version and _EXACT_SEMVER_RE.match(version):
        return McpPinning("pinned", f"dnx: exact version {version!r} of {spec!r}")
    if version and _is_floating_tag(version, composite=False):
        return McpPinning("unpinned", f"dnx: package {spec!r} uses the floating version {version!r}", floating=True)
    return McpPinning("unpinned", f"dnx: package {spec!r} has no exact version (resolves to the latest release)")


def _container_run_image(args: list[str]) -> str | None | bool:
    """The image of ``docker [global opts] run|container run [opts] IMAGE``; ``False`` when it is not a run."""
    sub, rest = _subcommand(args, _CONTAINER_GLOBAL_VALUE_FLAGS)
    if sub == "container" and rest[:1] == ["run"]:
        rest = rest[1:]
    elif sub != "run":
        return False
    return _first_positional(rest, _DOCKER_VALUE_FLAGS)


def mcp_container_image(config: Any) -> str | None:
    """Return the image reference a ``docker|podman|nerdctl run`` MCP server launches, if any.

    Uses the same argv parsing as :func:`classify_mcp_pinning` (wrappers such as
    ``cmd /c`` are looked through), so a flag value is never mistaken for the
    image. Returns ``None`` for every other server kind.
    """
    if not isinstance(config, dict):
        return None
    command = config.get("command")
    if not isinstance(command, str) or not command.strip():
        return None
    raw_args = config.get("args")
    args = [str(arg) for arg in raw_args] if isinstance(raw_args, list) else []
    argv = unwrap_launch_command(_launch_argv(command, args))
    if not argv or _command_basename(argv[0]) not in _CONTAINER_RUNTIMES:
        return None
    image = _container_run_image(argv[1:])
    return image if isinstance(image, str) else None


def is_exact_container_image(image: str) -> bool:
    """True when an image reference names one immutable (digest) or exact-version (tag) image."""
    return classify_image_pinning(image).status == "pinned"


def classify_image_pinning(image: str) -> McpPinning:
    """Public wrapper around the container-image pinning classifier."""
    return _classify_image(image.strip())


def _classify_npm_runner(runner: str, tokens: list[str], value_flags: frozenset[str]) -> McpPinning:
    # '-p a@latest,b' is read as two packages, so a floating one cannot hide behind a comma.
    packages = [part for value in _flag_values(tokens, ("-p", "--package")) for part in value.split(",") if part]
    if packages:
        return _prefixed(f"{runner}: ", _classify_spec_list(packages, _classify_npm_spec))
    spec = _first_positional(tokens, value_flags)
    if spec is None:
        return McpPinning("not_applicable", f"{runner} invocation without a package spec")
    return _prefixed(f"{runner}: ", _classify_npm_spec(spec))


def _prefixed(prefix: str, pin: McpPinning) -> McpPinning:
    return McpPinning(pin.status, f"{prefix}{pin.detail}", pin.floating)


def _redacted_pin(pin: McpPinning) -> McpPinning:
    """The classification with credentials in its detail removed (a spec can be a URL with a token in it)."""
    return McpPinning(pin.status, redact_secrets(pin.detail), pin.floating)


# --------------------------------------------------------------------------- #
# Network-free endpoint policy                                                #
# --------------------------------------------------------------------------- #
EndpointKind = Literal["metadata", "private"]

_METADATA_HOSTNAMES = frozenset(
    {
        "metadata.google.internal",
        "metadata",
        "instance-data",  # AWS (EC2 resolves it to 169.254.169.254)
        "instance-data.ec2.internal",
        "metadata.tencentyun.com",  # Tencent Cloud
    }
)
_METADATA_ADDRESSES = frozenset(
    {
        ipaddress.ip_address("169.254.169.254"),  # AWS, Azure, GCP, OpenStack, and most other clouds
        ipaddress.ip_address("fd00:ec2::254"),  # AWS IMDS over IPv6
        ipaddress.ip_address("100.100.100.200"),  # Alibaba Cloud
        ipaddress.ip_address("169.254.0.23"),  # Tencent Cloud
        ipaddress.ip_address("168.63.129.16"),  # Azure host (WireServer): extension settings and certificates
        ipaddress.ip_address("192.0.0.192"),  # Oracle Cloud (legacy)
        ipaddress.ip_address("169.254.170.2"),  # AWS ECS task credentials
        ipaddress.ip_address("169.254.170.23"),  # AWS EKS Pod Identity credentials
        ipaddress.ip_address("fd00:ec2::23"),  # AWS EKS Pod Identity credentials over IPv6
    }
)
_LOOPBACK_HOSTNAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"})
_PRIVATE_NETWORKS: tuple[tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, str], ...] = (
    (ipaddress.ip_network("127.0.0.0/8"), "loopback"),
    (ipaddress.ip_network("0.0.0.0/8"), "unspecified / this-network"),
    (ipaddress.ip_network("10.0.0.0/8"), "private (RFC 1918)"),
    (ipaddress.ip_network("172.16.0.0/12"), "private (RFC 1918)"),
    (ipaddress.ip_network("192.168.0.0/16"), "private (RFC 1918)"),
    (ipaddress.ip_network("100.64.0.0/10"), "carrier-grade NAT (100.64.0.0/10)"),
    (ipaddress.ip_network("169.254.0.0/16"), "link-local"),
    (ipaddress.ip_network("192.0.0.0/24"), "special-purpose (IETF protocol assignments)"),
    (ipaddress.ip_network("192.0.2.0/24"), "special-purpose (documentation)"),
    (ipaddress.ip_network("198.51.100.0/24"), "special-purpose (documentation)"),
    (ipaddress.ip_network("203.0.113.0/24"), "special-purpose (documentation)"),
    (ipaddress.ip_network("224.0.0.0/4"), "multicast"),
    (ipaddress.ip_network("240.0.0.0/4"), "special-purpose (reserved, including broadcast)"),
    (ipaddress.ip_network("::1/128"), "loopback"),
    (ipaddress.ip_network("::/128"), "unspecified"),
    (ipaddress.ip_network("fc00::/7"), "unique local (fc00::/7)"),
    (ipaddress.ip_network("fe80::/10"), "link-local"),
    (ipaddress.ip_network("fec0::/10"), "site-local (deprecated)"),
    (ipaddress.ip_network("2001:db8::/32"), "special-purpose (documentation)"),
    (ipaddress.ip_network("ff00::/8"), "multicast"),
)
# 198.18.0.0/15 (benchmarking) is not globally routable, but fake-IP proxy and VPN
# tools answer every DNS query from it and forward the connection to the real,
# public host. Classifying it as private would flag every endpoint on such a
# machine, so it is treated like a public address.
_FAKE_IP_NETWORK = ipaddress.ip_network("198.18.0.0/15")
_NAT64_NETWORK = ipaddress.ip_network("64:ff9b::/96")
# Ideographic / fullwidth / halfwidth full stops that UTS #46 maps to '.'.
_DOT_LOOKALIKES = str.maketrans({chr(0x3002): ".", chr(0xFF0E): ".", chr(0xFF61): "."})
_ENDPOINT_STATIC_NOTE = "static check only: DNS resolution and HTTP redirects are not evaluated"

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


@dataclass(frozen=True)
class EndpointClass:
    """A non-public MCP endpoint host found without any network access."""

    kind: EndpointKind
    reason: str
    host: str
    address: IPAddress | None = None
    encoded: bool = False


def _bare_host(host: str) -> str:
    """Lower-case host without brackets, an IPv6 zone ID, or a trailing dot (nothing decoded)."""
    text = host.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if ":" in text:
        text = text.split("%", 1)[0]  # IPv6 zone ID, e.g. fe80::1%25eth0
    return text.rstrip(".").lower()


def _normalize_host(host: str) -> str:
    text = _bare_host(host)
    if "%" in text:
        # WHATWG URL parsing (Node, MCP clients) percent-decodes a non-IPv6 host
        # before its IDNA and IPv4 steps: %31%32%37.0.0.1 is 127.0.0.1.
        text = unquote(text)
    text = text.translate(_DOT_LOOKALIKES).rstrip(".").lower()
    if not text.isascii():
        try:
            text = idna.encode(text, uts46=True).decode("ascii").lower()
        except idna.IDNAError:
            text = unicodedata.normalize("NFKC", text).lower()
    return text


def _parse_legacy_ipv4(text: str) -> ipaddress.IPv4Address | None:
    """Parse inet_aton-style IPv4 (decimal/hex/octal parts, 1-4 components)."""
    parts = text.split(".")
    if not 1 <= len(parts) <= 4 or any(not part for part in parts):
        return None
    numbers: list[int] = []
    for part in parts:
        lowered = part.lower()
        try:
            if lowered.startswith("0x"):
                if len(lowered) == 2:
                    return None
                numbers.append(int(lowered, 0))
            elif len(lowered) > 1 and lowered.startswith("0"):
                numbers.append(int(lowered, 8))
            elif lowered.isdigit():
                numbers.append(int(lowered, 10))
            else:
                return None
        except ValueError:
            return None
    *head, last = numbers
    if any(number > 0xFF for number in head) or last >= 1 << (8 * (4 - len(head))):
        return None
    value = 0
    for number in head:
        value = (value << 8) | number
    value = (value << (8 * (4 - len(head)))) | last
    return ipaddress.IPv4Address(value)


def _parse_host_address(host: str) -> tuple[IPAddress | None, bool]:
    """Return ``(address, encoded)`` for an IP-literal host (``None`` for names)."""
    try:
        return ipaddress.ip_address(host), False
    except ValueError:
        pass
    legacy = _parse_legacy_ipv4(host)
    if legacy is not None:
        return legacy, str(legacy) != host
    return None, False


def _embedded_ipv4(address: IPAddress) -> ipaddress.IPv4Address | None:
    """Return an IPv4 address embedded in an IPv6 literal (mapped/6to4/Teredo/NAT64/compat)."""
    if not isinstance(address, ipaddress.IPv6Address):
        return None
    if address.ipv4_mapped is not None:
        return address.ipv4_mapped
    if address.sixtofour is not None:
        return address.sixtofour
    if address.teredo is not None:
        return address.teredo[1]
    if address in _NAT64_NETWORK:
        return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
    value = int(address)
    if value >> 32 == 0 and value > 1:  # deprecated IPv4-compatible ::a.b.c.d
        return ipaddress.IPv4Address(value)
    return None


def _address_class(address: IPAddress) -> tuple[EndpointKind, str] | None:
    if address in _METADATA_ADDRESSES:
        return "metadata", "cloud instance-metadata"
    for network, reason in _PRIVATE_NETWORKS:
        if address.version == network.version and address in network:
            return "private", reason
    return None


def _not_globally_reachable(address: IPAddress) -> tuple[EndpointKind, str] | None:
    """Any other special-purpose address (IANA: not globally reachable, multicast, or reserved) is not public."""
    if address.version == 4 and address in _FAKE_IP_NETWORK:
        return None
    if address.is_multicast:
        return "private", "multicast"
    if not address.is_global or address.is_reserved:
        return "private", "special-purpose (not globally reachable)"
    return None


def _classify_address(address: IPAddress) -> tuple[tuple[EndpointKind, str] | None, bool]:
    """``(classification, embedded)``: an embedded IPv4 (mapped, 6to4, Teredo, NAT64, compatible) decides for it."""
    found = _address_class(address)
    if found is not None:
        return found, False
    inner = _embedded_ipv4(address)
    if inner is not None:
        return _address_class(inner) or _not_globally_reachable(inner), True
    return _not_globally_reachable(address), False


def classify_endpoint_address(address: IPAddress) -> tuple[EndpointKind, str] | None:
    """Classify one resolved IP address as ``metadata`` or ``private`` (``None`` when public).

    IPv4 addresses embedded in IPv6 (mapped, 6to4, Teredo, NAT64, compatible)
    are classified by the embedded address, like IP-literal hosts. Any address
    that is not globally reachable (special-purpose, documentation, reserved,
    broadcast, or multicast) counts as ``private``, except 198.18.0.0/15, which
    fake-IP proxy tools hand out for public names.
    """
    return _classify_address(address)[0]


def endpoint_client_host(host: str) -> str:
    """The host name a WHATWG client (Node, the MCP SDKs) looks up for ``host``.

    Percent-encoding is decoded and the name is lower-cased and IDNA-mapped, the
    way the URL parser does before it connects, so ``internal%2eexample`` is
    looked up as ``internal.example``. Brackets and a trailing dot are dropped.
    """
    return _normalize_host(host)


def classify_endpoint_host(host: str) -> EndpointClass | None:
    """Classify an MCP URL host as a metadata or private endpoint, network-free.

    Only IP literals (including IPv4-mapped/embedded IPv6 and decimal, hex, or
    octal IPv4 encodings) and well-known names are recognized, after the same
    percent-decoding a WHATWG URL parser applies to the host. A public-looking
    hostname returns ``None``: DNS resolution and redirect targets are not checked
    statically, so a public name may still resolve to a private address.
    """
    normalized = _normalize_host(host)
    if not normalized:
        return None
    # Percent-encoding, fullwidth digits, lookalike dots, or IDNA forms.
    name_encoded = normalized != _bare_host(host)
    if normalized in _METADATA_HOSTNAMES:
        return EndpointClass("metadata", "cloud instance-metadata", normalized, encoded=name_encoded)
    if normalized in _LOOPBACK_HOSTNAMES or normalized.endswith(".localhost"):
        return EndpointClass("private", "loopback", normalized, encoded=name_encoded)
    address, encoded = _parse_host_address(normalized)
    if address is None:
        return None
    found, embedded = _classify_address(address)
    encoded = encoded or name_encoded or embedded
    if found is None:
        return None
    kind, reason = found
    return EndpointClass(kind, reason, normalized, address, encoded)


def host_is_allowlisted(endpoint: EndpointClass, allowed_hosts: Iterable[str]) -> bool:
    """True when a policy entry allows this private host.

    Entries are exact host names, ``*.suffix`` wildcards, IP literals, or CIDR
    networks (e.g. ``10.0.0.0/8``). Cloud metadata endpoints are never allowlisted.
    """
    if endpoint.kind == "metadata":
        return False
    candidates: list[IPAddress] = []
    if endpoint.address is not None:
        candidates.append(endpoint.address)
        inner = _embedded_ipv4(endpoint.address)
        if inner is not None:
            candidates.append(inner)
    for raw in allowed_hosts:
        if not isinstance(raw, str):
            continue
        entry = _normalize_host(raw)
        if not entry:
            continue
        if entry.startswith("*.") and endpoint.host.endswith(entry[1:]):
            return True
        if entry == endpoint.host:
            return True
        try:
            network = ipaddress.ip_network(entry, strict=False)
        except ValueError:
            continue
        if any(candidate.version == network.version and candidate in network for candidate in candidates):
            return True
    return False


def host_name_is_allowlisted(host: str, allowed_hosts: Iterable[str]) -> bool:
    """True when a policy entry names this host (exact name or ``*.suffix``).

    Only host names match here: IP literals and CIDR entries are checked against
    resolved addresses with :func:`host_is_allowlisted`, and cloud metadata host
    names are never allowlisted.
    """
    normalized = _normalize_host(host)
    if not normalized or normalized in _METADATA_HOSTNAMES:
        return False
    address, _encoded = _parse_host_address(normalized)
    if address is not None:
        return False
    for raw in allowed_hosts:
        if not isinstance(raw, str):
            continue
        entry = _normalize_host(raw)
        if entry and (entry == normalized or (entry.startswith("*.") and normalized.endswith(entry[1:]))):
            return True
    return False


def _validate_endpoint(
    name: str,
    url: str,
    host: str | None,
    file_path: str,
    findings: list[Finding],
    allowed_private_hosts: Iterable[str],
    *,
    label: str = "url",
    shown: str | None = None,
) -> None:
    if not host:
        return
    endpoint = classify_endpoint_host(host)
    if endpoint is None:
        return
    encoded = f" (encoded as {host!r})" if endpoint.encoded else ""
    shown = shown if shown is not None else redacted_url(url)  # never echo userinfo or query credentials
    if endpoint.kind == "metadata":
        findings.append(
            _finding(
                Severity.HIGH,
                "mcp_endpoint_metadata",
                f"{label} targets a {endpoint.reason} endpoint{encoded}: {shown!r}; an MCP client pointed here "
                f"can expose instance credentials ({_ENDPOINT_STATIC_NOTE})",
                file_path,
                "Remove the instance-metadata endpoint; MCP servers must never target cloud metadata services.",
                name=name,
            )
        )
        return
    if host_is_allowlisted(endpoint, allowed_private_hosts):
        return
    findings.append(
        _finding(
            Severity.MEDIUM,
            "mcp_endpoint_private",
            f"{label} host is a {endpoint.reason} address{encoded}: {shown!r}; the endpoint is not publicly "
            f"reachable and may target local services ({_ENDPOINT_STATIC_NOTE})",
            file_path,
            "Use a public HTTPS endpoint, or allow this intended private host through the validation policy "
            "(mcp.allowed_private_hosts).",
            name=name,
        )
    )


# --------------------------------------------------------------------------- #
# Permission-bypass flags and dangerous overrides                             #
# --------------------------------------------------------------------------- #
PERMISSION_BYPASS_FLAGS: tuple[str, ...] = (
    "--dangerously-skip-permissions",
    "--dangerously-bypass-approvals-and-sandbox",
    "--dangerously-bypass-hook-trust",
    "--allow-dangerously-skip-permissions",
    "--yolo",
)
_BYPASS_FLAG_RE = re.compile(
    r"(?<![\w-])(" + "|".join(re.escape(flag) for flag in PERMISSION_BYPASS_FLAGS) + r")(?![\w-])",
    re.IGNORECASE,
)
# Option/value pairs with the same effect: Claude Code's permission mode, Gemini
# CLI's approval mode, and Codex CLI's sandbox and approval policy (long and short option).
PERMISSION_BYPASS_OPTIONS: tuple[tuple[str, str], ...] = (
    ("--permission-mode", "bypassPermissions"),
    ("--approval-mode", "yolo"),
    ("--sandbox", "danger-full-access"),
    ("-s", "danger-full-access"),
    ("--ask-for-approval", "never"),
    ("-a", "never"),
)
# Codex CLI config overrides with the same effect: '-c approval_policy=never',
# '--config sandbox_mode="danger-full-access"'.
PERMISSION_BYPASS_CONFIG: tuple[tuple[str, str], ...] = (
    ("approval_policy", "never"),
    ("sandbox_mode", "danger-full-access"),
)
# '-a', '-s', and '-c' / '--config' are common options, so '-a never', '-s danger-full-access', and the config
# overrides count only after a codex command in the same shell command or argv ('grep -a never f' and
# 'tool -s danger-full-access' are not Codex). A codex command may be a wrapper named for it ('codex-wrapper',
# 'my-codex') or a variable ('${CODEX_BIN}').
_CODEX_ONLY_OPTIONS = frozenset({"-a", "-s"})
_CODEX_COMMAND_RE = re.compile(
    r"(?<![\w.-])(?:[\w.-]*[-_.])?codex(?:[-_][\w-]*)?(?:\.exe|\.cmd)?(?![\w.])|\$\{?CODEX\w*", re.IGNORECASE
)
# Gemini CLI's '-y' is short for '--yolo'; '-y' alone means "yes" to many tools (npx, apt), so it counts only
# after a gemini command in the same shell command or argv.
_GEMINI_COMMAND_RE = re.compile(
    r"(?<![\w.-])(?:[\w.-]*[-_.])?gemini(?:[-_][\w-]*)?(?:\.exe|\.cmd)?(?![\w.])", re.IGNORECASE
)
_GEMINI_YES_RE = re.compile(r"(?<![\w-])-y(?![\w-])")
# Where one shell command ends: a newline, ';', '&&', '||', a pipe, or a lone '&' (a background job). The '&'
# or '|' of a redirection ('2>&1', '>&2', '&>log', '<&3', '>|log') does not end the command.
_COMMAND_BREAK_RE = re.compile(r"\|\||&&|[;\n]|(?<!>)\||(?<![<>&])&(?![>&])")
# A backslash-newline joins two lines into one shell command ('codex exec \\\n  -a never').
_LINE_CONTINUATION_RE = re.compile(r"\\\r?\n")
_BYPASS_CONFIG_OPTIONS = frozenset({"-c", "--config"})
_BYPASS_CONFIG_VALUE_RES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (f"-c {key}={value}", re.compile(rf"^[\"']?{key}\s*=\s*[\"']?{re.escape(value)}[\"']?$", re.IGNORECASE))
    for key, value in PERMISSION_BYPASS_CONFIG
)
# In one string: "--opt value", "--opt=value", or a quoted value; the last item is True for a Codex-only form.
_BYPASS_OPTION_RES: tuple[tuple[str, re.Pattern[str], bool], ...] = (
    *(
        (
            f"{option} {value}",
            re.compile(rf"(?<![\w-]){re.escape(option)}(?:=|\s+)[\"']?{re.escape(value)}(?![\w-])", re.IGNORECASE),
            option in _CODEX_ONLY_OPTIONS,
        )
        for option, value in PERMISSION_BYPASS_OPTIONS
    ),
    *(
        (
            f"-c {key}={value}",
            re.compile(
                rf"(?<![\w-])(?:-c|--config)(?:=|\s+)[\"']?{key}\s*=\s*[\"']?{re.escape(value)}(?![\w-])",
                re.IGNORECASE,
            ),
            True,
        )
        for key, value in PERMISSION_BYPASS_CONFIG
    ),
)
# Split across adjacent argv tokens: option -> (value, reported flag, Codex-only).
_BYPASS_OPTION_VALUES: dict[str, tuple[str, str, bool]] = {
    option.lower(): (value.lower(), f"{option} {value}", option in _CODEX_ONLY_OPTIONS)
    for option, value in PERMISSION_BYPASS_OPTIONS
}
# Keys whose values are prose, never executed config -- documentation mentions of
# a flag are not flagged.
_DOC_KEYS = frozenset({"description", "title", "summary", "notes", "note", "comment", "comments", "help"})
# Matches the load_bounded_json node limit, so every config it accepts is walked
# completely; a walk that still hits the bound is reported, never silently cut.
_MAX_SCAN_NODES = MAX_STRUCTURED_NODES
_CODE_INJECTION_ENV = frozenset({"LD_PRELOAD", "LD_AUDIT", "DYLD_INSERT_LIBRARIES"})
# BASH_ENV names a file every non-interactive bash runs first; PERL5OPT and RUBYOPT can load a module.
_STARTUP_FILE_ENV = frozenset({"BASH_ENV"})
_INTERPRETER_OPTION_ENV: dict[str, re.Pattern[str]] = {
    "PERL5OPT": re.compile(r"(?:^|\s)-[MmI]"),
    "RUBYOPT": re.compile(r"(?:^|\s)-[rI]"),
}
# Module search paths: a path outside the plugin can shadow the modules the launched process imports.
_MODULE_PATH_ENV = frozenset({"PYTHONPATH", "NODE_PATH", "PERL5LIB", "PERLLIB", "RUBYLIB"})
_PLUGIN_ROOT_PATH_RE = re.compile(r"^\$\{?(?:CLAUDE_PLUGIN_ROOT|PLUGIN_ROOT|CURSOR_PLUGIN_ROOT)\}?(?:[/\\]|$)")
_NODE_OPTIONS_INJECTION_RE = re.compile(
    r"(?:^|\s)(--require|-r|--import|--loader|--experimental-loader)(?:=|\s|$)", re.IGNORECASE
)
_TRAFFIC_REDIRECT_ENV = frozenset(
    {
        "ANTHROPIC_BASE_URL",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "AZURE_OPENAI_ENDPOINT",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NODE_EXTRA_CA_CERTS",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "CURL_CA_BUNDLE",
    }
)
_LLM_PROVIDER_BASE_URL_RE = re.compile(
    r"^(?:ANTHROPIC|CLAUDE|OPENAI|AZURE_OPENAI|AZURE_AI|GOOGLE|GEMINI|VERTEX|VERTEX_AI|MISTRAL|COHERE|GROQ|"
    r"TOGETHER|OPENROUTER|DEEPSEEK|XAI|HF|HUGGINGFACE|OLLAMA|BEDROCK|LITELLM|PERPLEXITY|FIREWORKS)"
    r"(?:_[A-Z0-9]+)*_(?:BASE_URL|API_BASE|API_BASE_URL)$"
)
_AUTO_APPROVE_KEYS = frozenset({"autoapprove", "alwaysallow", "autoapprovetools", "alwaysallowtools"})


@dataclass(frozen=True)
class OverrideIssue:
    """A dangerous flag/env/config override, independent of where it was declared."""

    concept: Literal[
        "permission_bypass_flag",
        "permission_bypass_scan_truncated",
        "permission_mode_flag",
        "permission_allow_flag",
        "env_code_injection",
        "env_traffic_redirect",
        "env_insecure_tls",
        "env_inline_secret",
        "auto_approve",
    ]
    severity: Severity
    message: str
    suggestion: str


class _ConfigWalk:
    """Bounded, iterative walk yielding ``(json_path, node)`` for a config value.

    ``complete`` turns False when the walk stops at ``_MAX_SCAN_NODES`` with nodes
    still unvisited.
    """

    def __init__(self, value: Any, *, skip_doc_keys: bool = True) -> None:
        self.value = value
        self.skip_doc_keys = skip_doc_keys
        self.complete = True

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        stack: list[tuple[str, Any]] = [("", self.value)]
        seen = 0
        while stack:
            if seen >= _MAX_SCAN_NODES:
                self.complete = False
                return
            path, node = stack.pop()
            seen += 1
            yield path, node
            if isinstance(node, dict):
                for key, child in node.items():
                    if self.skip_doc_keys and str(key).lower() in _DOC_KEYS:
                        continue
                    stack.append((f"{path}.{key}" if path else str(key), child))
            elif isinstance(node, list):
                for index, child in enumerate(node):
                    stack.append((f"{path}[{index}]", child))


def iter_config_strings(value: Any, *, skip_doc_keys: bool = True) -> Iterator[tuple[str, str]]:
    """Yield ``(json_path, string)`` for string leaves of a config value (bounded, iterative)."""
    for path, node in _ConfigWalk(value, skip_doc_keys=skip_doc_keys):
        if isinstance(node, str):
            yield path, node


def _join_continuations(text: str) -> str:
    """``text`` with each backslash-newline replaced by spaces of the same length (offsets stay the same)."""
    return _LINE_CONTINUATION_RE.sub(lambda match: " " * len(match.group()), text)


def _command_before(text: str, start: int, command_re: re.Pattern[str]) -> bool:
    """Whether ``command_re`` matches before ``start`` in the same shell command of ``text``.

    ``echo codex done; grep -a never f`` puts the codex word in another command,
    so it does not make ``-a never`` a Codex option. A backslash-continued line and
    a redirection (``codex exec 2>&1 -a never``) stay in the same command.
    """
    text = _join_continuations(text)
    begin = 0
    for match in _COMMAND_BREAK_RE.finditer(text, 0, start):
        begin = match.end()
    return command_re.search(text, begin, start) is not None


def _codex_hit(text: str, start: int) -> bool:
    """Whether a codex command comes before ``start`` in the same shell command of ``text``."""
    return _command_before(text, start, _CODEX_COMMAND_RE)


def _bypass_hits(path: str, node: Any) -> Iterator[tuple[str, str]]:
    """Yield ``(json_path, flag)`` for bypass flags in a string or split across argv tokens."""
    if isinstance(node, str):
        node = _join_continuations(node)
        for match in _BYPASS_FLAG_RE.finditer(node):
            yield path, match.group(1).lower()
        for label, pattern, codex_only in _BYPASS_OPTION_RES:
            if any(not codex_only or _codex_hit(node, hit.start()) for hit in pattern.finditer(node)):
                yield path, label
        if any(_command_before(node, hit.start(), _GEMINI_COMMAND_RE) for hit in _GEMINI_YES_RE.finditer(node)):
            yield path, "gemini -y"
    elif isinstance(node, list):
        yield from _argv_bypass_hits(path, node, codex=False)
    elif isinstance(node, dict):
        # {"command": "codex", "args": ["-a", "never"]}: the args follow a codex (or gemini) command.
        command, args = node.get("command"), node.get("args")
        if isinstance(command, str) and isinstance(args, list):
            args_path = f"{path}.args" if path else "args"
            if _CODEX_COMMAND_RE.search(command):
                yield from _argv_bypass_hits(args_path, args, codex=True)
            if _GEMINI_COMMAND_RE.search(command):
                yield from _argv_bypass_hits(args_path, args, codex=False, gemini=True)


def _argv_bypass_hits(path: str, argv: list[Any], *, codex: bool, gemini: bool = False) -> Iterator[tuple[str, str]]:
    """Bypass options split across adjacent argv tokens (``["--sandbox", "danger-full-access"]``); the
    Codex-only forms count only after a codex token, or when ``codex`` says the argv belongs to one, and
    Gemini CLI's ``-y`` only after a gemini token (or when ``gemini`` says so)."""
    for index, token in enumerate(argv):
        if isinstance(token, str) and _GEMINI_COMMAND_RE.search(token):
            gemini = True
        elif gemini and isinstance(token, str) and token.strip() == "-y":
            yield f"{path}[{index}]", "gemini -y"
    for index, (option, value) in enumerate(itertools.pairwise(argv)):
        if isinstance(option, str) and _CODEX_COMMAND_RE.search(option):
            codex = True
        if not isinstance(option, str) or not isinstance(value, str):
            continue
        expected = _BYPASS_OPTION_VALUES.get(option.strip().lower())
        if expected is not None and value.strip().strip("\"'").lower() == expected[0] and (codex or not expected[2]):
            yield f"{path}[{index}]", expected[1]
        if codex and option.strip().lower() in _BYPASS_CONFIG_OPTIONS:
            for label, pattern in _BYPASS_CONFIG_VALUE_RES:
                if pattern.match(value.strip()):
                    yield f"{path}[{index}]", label


def permission_bypass_issues(value: Any) -> list[OverrideIssue]:
    """Find agent-CLI permission-bypass flags in any config/command string or argv list.

    A config too large to walk completely is itself a HIGH issue (fail closed).
    """
    issues: list[OverrideIssue] = []
    seen: set[tuple[str, str]] = set()
    walk = _ConfigWalk(value)
    for node_path, node in walk:
        for path, flag in _bypass_hits(node_path, node):
            if (path, flag) in seen:
                continue
            seen.add((path, flag))
            where = f" in '{path}'" if path else ""
            issues.append(
                OverrideIssue(
                    "permission_bypass_flag",
                    Severity.HIGH,
                    f"agent-CLI permission-bypass flag {flag!r}{where} disables tool-approval prompts or sandboxing",
                    "Remove the permission-bypass flag; plugins must not disable the host agent's approvals or sandbox.",
                )
            )
    if not walk.complete:
        issues.append(
            OverrideIssue(
                "permission_bypass_scan_truncated",
                Severity.HIGH,
                f"config has more than {_MAX_SCAN_NODES} nodes; the permission-bypass scan stopped before "
                "inspecting all of it",
                "Split or simplify the config so every entry can be inspected.",
            )
        )
    return issues


def _is_passthrough(key: str, value: str) -> bool:
    """True for an unset value or a pure ``${KEY}`` passthrough of the same variable."""
    stripped = value.strip()
    if not stripped:
        return True
    return stripped in {f"${{{key}}}", f"${key}"}


def env_override_issues(env: Any) -> list[OverrideIssue]:
    """Find env overrides that inject code or redirect model/API traffic (values are never echoed)."""
    if not isinstance(env, dict):
        return []
    issues: list[OverrideIssue] = []
    for raw_key, raw_value in env.items():
        key = str(raw_key).strip()
        upper = key.upper()
        value = raw_value if isinstance(raw_value, str) else ("" if raw_value is None else str(raw_value))
        if _is_passthrough(key, value):
            continue
        if upper in _CODE_INJECTION_ENV:
            issues.append(
                OverrideIssue(
                    "env_code_injection",
                    Severity.HIGH,
                    f"env '{key}' preloads a shared library into the launched process (code injection)",
                    f"Remove '{key}'; plugins must not inject libraries into the processes they launch.",
                )
            )
        elif upper in _STARTUP_FILE_ENV:
            issues.append(
                OverrideIssue(
                    "env_code_injection",
                    Severity.HIGH,
                    f"env '{key}' makes every non-interactive bash in the launched process run a file first "
                    "(code injection)",
                    f"Remove '{key}'; plugins must not inject startup code into the processes they launch.",
                )
            )
        elif upper in _INTERPRETER_OPTION_ENV and _INTERPRETER_OPTION_ENV[upper].search(value):
            issues.append(
                OverrideIssue(
                    "env_code_injection",
                    Severity.HIGH,
                    f"env '{key}' loads extra code or a module search path into every {upper[:4].lower()} process",
                    f"Remove '{key}'; load code explicitly from the server instead.",
                )
            )
        elif upper in _MODULE_PATH_ENV and not all(
            not entry.strip()
            or _PLUGIN_ROOT_PATH_RE.match(entry.strip().strip("\"'"))
            or _is_passthrough(key, entry.strip().strip("\"'"))
            for entry in re.split(r"[:;]", value)
        ):
            issues.append(
                OverrideIssue(
                    "env_code_injection",
                    Severity.MEDIUM,
                    f"env '{key}' adds module search paths outside the plugin root, which can shadow the modules "
                    "the launched process imports",
                    f"Point '{key}' only at the plugin's own files (${{CLAUDE_PLUGIN_ROOT}}/...), or remove it.",
                )
            )
        elif upper == "NODE_OPTIONS" and (match := _NODE_OPTIONS_INJECTION_RE.search(value)):
            issues.append(
                OverrideIssue(
                    "env_code_injection",
                    Severity.HIGH,
                    f"env 'NODE_OPTIONS' uses {match.group(1)!r} to load extra code into every Node.js process",
                    "Remove --require/--import/--loader from NODE_OPTIONS; load code explicitly from the server.",
                )
            )
        elif upper in _TRAFFIC_REDIRECT_ENV or _LLM_PROVIDER_BASE_URL_RE.match(upper):
            issues.append(
                OverrideIssue(
                    "env_traffic_redirect",
                    Severity.MEDIUM,
                    f"env '{key}' overrides model/API endpoints, proxies, or trusted CA certificates, which can "
                    "redirect or intercept traffic",
                    f"Remove '{key}' or pass the user's own value through as \"${{{key}}}\"; do not ship "
                    "endpoint, proxy, or CA overrides.",
                )
            )
    return issues


def env_tls_and_secret_issues(env: Any, *, severity: Severity = Severity.CRITICAL) -> list[OverrideIssue]:
    """The MCP ``env`` checks for any other env block: TLS verification turned off, and inline credentials.

    The same rules as ``mcp_insecure_tls_env`` and ``mcp_inline_secret`` (values
    are never echoed). LSP servers get them at CRITICAL like MCP servers; a
    settings file passes its own ``severity``.
    """
    if not isinstance(env, dict):
        return []
    issues: list[OverrideIssue] = []
    for raw_key, raw_value in env.items():
        key = str(raw_key).strip()
        if not isinstance(raw_value, str):
            continue
        if _is_insecure_tls_env(key, raw_value):
            issues.append(
                OverrideIssue(
                    "env_insecure_tls",
                    severity,
                    f"env '{key}' disables TLS/certificate verification in the launched process",
                    f"Remove '{key}'; do not disable TLS verification through environment variables.",
                )
            )
        if _looks_like_inline_secret(key, raw_value):
            issues.append(
                OverrideIssue(
                    "env_inline_secret",
                    severity,
                    f"env '{key}' contains an inline credential; only ${{ENV}} references are allowed",
                    'Reference the secret as an env var (e.g. "${MY_TOKEN}"); never ship a raw credential.',
                )
            )
    return issues


def auto_approve_issues(config: Any) -> list[OverrideIssue]:
    """Find MCP-entry keys that auto-approve tool calls without a user prompt."""
    if not isinstance(config, dict):
        return []
    issues: list[OverrideIssue] = []
    for raw_key, value in config.items():
        key = str(raw_key)
        normalized = key.lower().replace("_", "").replace("-", "")
        flagged = (normalized in _AUTO_APPROVE_KEYS and bool(value)) or (normalized == "trust" and value is True)
        if flagged:
            issues.append(
                OverrideIssue(
                    "auto_approve",
                    Severity.MEDIUM,
                    f"'{key}' auto-approves MCP tool calls without a user prompt",
                    f"Remove '{key}'; let the host agent prompt for tool approval.",
                )
            )
    return issues


def _append_override_findings(name: str, issues: list[OverrideIssue], file_path: str, findings: list[Finding]) -> None:
    for issue in issues:
        check = "mcp_auto_approve" if issue.concept == "auto_approve" else f"mcp_{issue.concept}"
        findings.append(_finding(issue.severity, check, issue.message, file_path, issue.suggestion, name=name))


def _validate_overrides(name: str, config: dict[str, Any], file_path: str, findings: list[Finding]) -> None:
    """Permission-bypass flags anywhere in the entry, permissive agent-CLI flags (``--permission-mode
    acceptEdits``, ``--allowedTools Bash``), dangerous env, and auto-approve keys."""
    from skillevaluator.plugin_component_risk import allowed_tools_flag_issues, permission_mode_flag_issues

    _append_override_findings(name, permission_bypass_issues(config), file_path, findings)
    _append_override_findings(
        name, [*permission_mode_flag_issues(config), *allowed_tools_flag_issues(config)], file_path, findings
    )
    _append_override_findings(name, env_override_issues(config.get("env")), file_path, findings)
    _append_override_findings(name, auto_approve_issues(config), file_path, findings)


def _validate_pinning(
    name: str, config: dict[str, Any], file_path: str, findings: list[Finding], *, client: McpClient = "claude"
) -> None:
    """HIGH for a moving tag (``@latest``, ``:main``) where the runner reads the version; MEDIUM for no exact version.

    The tag is read from the parsed package or image spec, never as a substring
    of any argument, so a scope (``@nextui-org/mcp@1.0.0``), an entry point
    (``mod.cli:main``), an e-mail, or a digest-pinned ``:latest@sha256:...`` is
    not floating. ``${NAME:-default}`` is read at its default except for Codex.
    """
    pin = classify_mcp_pinning(config, client=client)
    if pin.status != "unpinned":
        return
    if pin.floating:
        findings.append(
            _finding(
                Severity.HIGH,
                "mcp_command_floating_version",
                f"package runner uses a floating version or tag ({pin.detail}); each launch may fetch different code",
                file_path,
                "Pin the referenced package/image to an exact version or digest, not a moving tag such as latest "
                "or main.",
                name=name,
            )
        )
        return
    findings.append(
        _finding(
            Severity.MEDIUM,
            "mcp_unpinned_package",
            f"package runner is not pinned to an exact version ({pin.detail}); each launch may fetch different code",
            file_path,
            "Pin an exact version (pkg@1.2.3, pkg==1.2.3, --from pkg==1.2.3, image:1.2.3 or image@sha256:...).",
            name=name,
        )
    )


def _validate_cursor_references(
    name: str, config: dict[str, Any], file_path: str, findings: list[Finding], client: McpClient
) -> None:
    """LOW: Cursor's ``${env:NAME}`` is a reference (never an inline secret), but its expansion is unverified."""
    # Non-list 'args' is its own mcp_args_not_list finding; it must not crash this check.
    args = config.get("args")
    fields = [
        label
        for label, value in (
            ("command", config.get("command")),
            ("url", config.get("url")),
            *((f"args[{index}]", arg) for index, arg in enumerate(args if isinstance(args, list) else [])),
            *(
                (f"{section}.{key}", value)
                for section in ("env", "headers")
                if isinstance(config.get(section), dict)
                for key, value in config[section].items()
            ),
        )
        if isinstance(value, str) and "${env:" in value
    ]
    if not fields:
        return
    where = ", ".join(fields[:5]) + (f" and {len(fields) - 5} more" if len(fields) > 5 else "")
    expands = (
        "whether Cursor expands it in a plugin's MCP config is not verified"
        if client == "cursor"
        else "it is Cursor's form, and Claude Code and Codex are not documented to expand it"
    )
    findings.append(
        _finding(
            Severity.LOW,
            "mcp_env_reference_unverified",
            f"'${{env:NAME}}' in {where} is read as an environment reference, not an inline secret; {expands}",
            file_path,
            "Check that the client expands the reference; Claude Code expands ${NAME} and ${NAME:-default}.",
            name=name,
        )
    )


def validate_mcp_server_declaration(
    name: Any,
    config: Any,
    file_path: str,
    *,
    allowed_private_hosts: Iterable[str] = (),
    manifest_type: str | None = None,
) -> list[Finding]:
    """Statically validate one contained ``mcpServers`` entry (``name`` -> config).

    ``allowed_private_hosts`` comes from the validation policy
    (``mcp.allowed_private_hosts``) and suppresses ``mcp_endpoint_private`` for
    intended private hosts; cloud metadata endpoints are never allowlisted.
    ``manifest_type`` names the format whose client loads the server (Claude
    Code by default); it decides how ``${VAR}`` references in the URL expand.
    """
    client = mcp_client_for_manifest(manifest_type)
    findings: list[Finding] = []

    if not isinstance(name, str) or not _MCP_NAME_RE.match(name.strip()):
        findings.append(
            _finding(
                Severity.HIGH,
                "mcp_name_invalid",
                f"MCP server name {name!r} must start with an alphanumeric and use only letters, digits, '.', '_', '-'",
                file_path,
                "Rename the MCP server to a valid identifier.",
            )
        )
        # A non-string key cannot carry a config we can inspect further.
        if not isinstance(name, str):
            return findings

    if not isinstance(config, dict):
        findings.append(
            _finding(
                Severity.HIGH,
                "mcp_config_not_object",
                "MCP server config must be a JSON object",
                file_path,
                "Express the MCP server config as an object with command/url/provider.",
                name=name,
            )
        )
        return findings

    has_command = "command" in config
    has_url = "url" in config
    has_provider = "provider" in config
    declared_kinds = sum((has_command, has_url, has_provider))

    if declared_kinds == 0:
        findings.append(
            _finding(
                Severity.HIGH,
                "mcp_missing_kind",
                "MCP server must declare a 'command' (stdio), a 'url' (http/sse), or a 'provider'",
                file_path,
                "Add a runnable command/url, or declare a public provider identifier.",
                name=name,
            )
        )
    elif declared_kinds > 1:
        findings.append(
            _finding(
                Severity.HIGH,
                "mcp_kind_invalid",
                "MCP server must declare exactly one of 'command', 'url', or 'provider'",
                file_path,
                "Choose one runnable or provider-only MCP form.",
                name=name,
            )
        )

    _validate_transport(name, config, file_path, findings)
    _validate_insecure_tls_config(name, config, file_path, findings)
    _validate_env_and_headers(name, config, file_path, findings, client=client)
    _validate_overrides(name, config, file_path, findings)
    _validate_cursor_references(name, config, file_path, findings, client)

    if has_command:
        _validate_command(name, config, file_path, findings)
        _validate_pinning(name, config, file_path, findings, client=client)
    if has_url:
        _validate_url(name, config, file_path, findings, allowed_private_hosts, client=client)
    _validate_oauth_urls(name, config, file_path, findings, allowed_private_hosts, client=client)
    if has_provider and not (has_command or has_url):
        provider = config.get("provider")
        if not isinstance(provider, str) or not provider.strip():
            findings.append(
                _finding(
                    Severity.HIGH,
                    "mcp_provider_invalid",
                    "provider must be a non-empty string",
                    file_path,
                    "Set a public provider identifier.",
                    name=name,
                )
            )

    return findings


def validate_contained_mcp_servers(
    mcp_servers: Any,
    file_path: str,
    *,
    allowed_private_hosts: Iterable[str] = (),
    manifest_type: str | None = None,
) -> list[Finding]:
    """Statically validate an in-memory ``.claude-plugin/plugin.json`` ``mcpServers`` value.

    Accepts every documented Claude Code form without touching the filesystem:
    an inline server map, a path string, or an array mixing both. Inline maps
    (top-level or array elements) are validated here; path strings name JSON
    config files that :func:`skillevaluator.plugin_components.collect_mcp_declarations`
    reads through the bounded, no-follow plugin-root reader. Returns a (possibly
    empty) list of findings; an absent or empty value yields none.
    """
    if mcp_servers is None:
        return []
    if isinstance(mcp_servers, str):
        return []
    if isinstance(mcp_servers, list):
        findings: list[Finding] = []
        for index, entry in enumerate(mcp_servers):
            if isinstance(entry, str):
                continue
            if isinstance(entry, dict):
                findings.extend(
                    validate_contained_mcp_servers(
                        entry, file_path, allowed_private_hosts=allowed_private_hosts, manifest_type=manifest_type
                    )
                )
                continue
            findings.append(
                _finding(
                    Severity.HIGH,
                    "mcp_servers_entry_invalid",
                    f"mcpServers[{index}] must be a config-file path string or an inline server map "
                    f"(got {type(entry).__name__})",
                    file_path,
                    'Use "./path/to/servers.json" or {"<name>": {"command"|"url": ...}} for each array entry.',
                )
            )
        return findings
    if not isinstance(mcp_servers, dict):
        return [
            _finding(
                Severity.HIGH,
                "mcp_servers_not_object",
                "'mcpServers' must be an inline server map, a config-file path string, or an array of those "
                f"(got {type(mcp_servers).__name__})",
                file_path,
                'Express mcpServers as {"<name>": {"command"|"url"|"provider": ...}}, "./.mcp.json", '
                "or an array mixing both.",
            )
        ]
    findings = []
    for name, config in mcp_servers.items():
        findings.extend(
            validate_mcp_server_declaration(
                name, config, file_path, allowed_private_hosts=allowed_private_hosts, manifest_type=manifest_type
            )
        )
    return findings
