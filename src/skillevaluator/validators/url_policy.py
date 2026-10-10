# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""URL reading, credential checks, and report display shared by the MCP, hook, and endpoint policies.

Node (Claude Code http hooks, MCP SDK fetch), Rust's url crate, and browsers read
URLs with the WHATWG URL Standard, and urllib.parse does not always agree:
:func:`whatwg_url` gives the URL such a client connects to, and
:func:`url_ambiguities` says why the two readings could differ.

The credential predicates decide whether a keyed value (an env entry, a header,
a command-line flag, a URL query parameter) is an inline credential rather than
a reference (``$VAR``, ``${VAR}``, ``${VAR:-}``, or Cursor's ``${env:VAR}``);
:func:`url_credentials` applies them to a URL. MCP server URLs, HTTP hook URLs,
and URLs inside hook commands all use this one rule. :func:`safe_url`,
:func:`report_text`, and :func:`redact_secrets` are how URLs and other plugin
text appear in findings and reports, with credentials removed.

Nothing here touches the network.
"""

from __future__ import annotations

import bisect
import math
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
# Environment references. Claude Code expands '${NAME}' and '${NAME:-default}' in an MCP server's command,
# args, env, url, and headers (the default when NAME is unset); hooks and many configs also use '$NAME';
# Cursor documents '${env:NAME}'. Claude Code also fills '${user_config.KEY}' with the value the user gives for the
# plugin's userConfig option KEY. A reference with no default (or an empty one) carries no value of its own.
_ENV_EXPANSION_RE = re.compile(
    r"\$\{(?P<cursor>env:)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}"
    r"|\$\{user_config\.(?P<user_config>[A-Za-z0-9_.-]+)\}"
    r"|\$(?P<bare>[A-Za-z_][A-Za-z0-9_]*)"
)
# What is left of an Authorization-style value once its references are removed: 'Bearer ${TOKEN}'.
_AUTH_SCHEME_ONLY_RE = re.compile(r"(?i)^\s*(?:bearer|basic|token)?\s*$")
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
# The short prefixes 'sk-', 'hf_', and 'npm_' also end ordinary words and ids
# ('task-<hex>', 'disk-<hex>', 'pnpm_...'), so they match only where a word
# starts. The longer prefixes match anywhere, so a token glued to other text
# ('%3Dghp_...') is still found.
#
# The JWT-like alternative starts only where a run of token characters starts
# and scans to the run's first ``eyJ`` without ever stepping past one. A later
# ``eyJ`` in the same run has fewer characters before the run ends, so it can
# never match when the first one does not. A plain ``eyJ...`` alternative was
# tried at every ``eyJ`` and scanned to the end of the run each time, which is
# quadratic on a long ``eyJeyJ...`` value (about 1 s per 64 KB value).
_SECRET_VALUE_RE = re.compile(
    r"((?<![A-Za-z0-9_-])sk-[A-Za-z0-9]{16,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{22,}"
    r"|glpat-[A-Za-z0-9_-]{20,}"
    r"|(?:AKIA|ASIA)[0-9A-Z]{16}"
    r"|xox[abeprs]-[A-Za-z0-9-]{10,}"
    r"|nvapi-[A-Za-z0-9_-]{16,}"
    r"|(?<![A-Za-z0-9_-])hf_[A-Za-z0-9]{30,}"
    r"|(?<![A-Za-z0-9_-])npm_[A-Za-z0-9]{36}"
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
# Keys that name a credential in a URL query or an HTTP header (Google's 'key=', a signed URL's 'sig='), but
# are often a setting as a flag or env name ('--key primary', '--pass 2', '--signature sha256'). There the value
# must also look like secret material.
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


def is_env_reference(value: str) -> bool:
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


def has_secret_shape(value: str) -> bool:
    """True when a literal value is shaped like a known token or key, or is an inline ``Bearer``/``Basic`` credential.

    A reference (:func:`is_env_reference`) never is.
    """
    text = value.strip()
    if not text or is_env_reference(text):
        return False
    return _SECRET_VALUE_RE.search(text) is not None or _INLINE_AUTH_SCHEME_RE.match(text) is not None


def is_credential_key(key: str, *, query: bool = False) -> bool:
    """Whether a key names a credential (``API_KEY``, ``clientSecret``, ``--auth``), not a setting about one.

    A key whose last word makes it a setting (``PASSWORD_POLICY``,
    ``DB_PASSWORD_FILE``) does not. With ``query`` (a URL query parameter or an
    HTTP header name) the short keys ``key``, ``sig``, ``signature``, ``pass``,
    and ``pwd`` count too.
    """
    text = str(key).strip().lstrip("-")
    if text.lower() in _CREDENTIAL_KEYS_EXACT or (query and text.lower() in _QUERY_CREDENTIAL_KEYS):
        return True
    if not _SECRET_KEY_RE.search(text):
        return False
    words = _key_words(text)
    return not (words and words[-1] in _NON_CREDENTIAL_QUALIFIERS)


def looks_like_inline_secret(key: str, value: str, *, query: bool = True) -> bool:
    """True when a keyed value is an inline credential rather than a reference.

    A value shaped like a secret counts under any key, including an inline
    ``Bearer``/``Basic`` credential (so Authorization-style headers are covered
    without keying on the header name). A non-empty literal counts under a
    credential-named key (:func:`is_credential_key`) unless it is a setting: a
    boolean, an auth mode such as ``basic``, or a path. A short key such as
    ``KEY`` or ``--pass`` counts only with a value that looks like key material
    (:func:`looks_like_secret_material`). References never count, and in a value
    that mixes text and references (``Bearer ${TOKEN}``, ``${TOKEN:-default}``)
    only the text the plugin ships is judged.

    ``query`` says the key is a URL query parameter or an HTTP header name, where
    ``key=`` or ``sig=`` names a credential on its own; pass ``False`` for an env
    name or a command flag.
    """
    text = value.strip()
    if not text or is_env_reference(text):
        return False
    if _ENV_EXPANSION_RE.search(text):
        # Judge only what the plugin ships: the literal text around the references and their defaults.
        literal = env_defaults_text(text)
        if _AUTH_SCHEME_ONLY_RE.match(literal):
            return False
        return looks_like_inline_secret(key, literal, query=query)
    if has_secret_shape(text):
        return True
    if is_credential_key(str(key), query=query):
        return not _is_neutral_value(text)
    return _is_query_credential_key(str(key)) and not _is_neutral_value(text) and looks_like_secret_material(text)


def looks_like_secret_material(value: str) -> bool:
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
    return classes >= 3 or looks_like_random_secret("value", text)


def looks_like_random_secret(key: str, value: str) -> bool:
    """A long random-looking value (``SERVICE_SEED=9f2c...``) that may be key material whatever its key says.

    32 or more hex or base64 characters with high entropy. A UUID is not, and
    neither is a value under a key that names a hash, commit, version, or ID.
    """
    text = value.strip()
    if not _RANDOM_VALUE_RE.match(text) or _UUID_VALUE_RE.match(text):
        return False
    if _IDENTIFIER_KEY_WORDS & set(_key_words(key)):
        return False
    if _HEX_VALUE_RE.match(text):
        return _shannon_entropy(text) >= 3.0
    classes = sum(1 for pattern in (r"[a-z]", r"[A-Z]", r"[0-9]") if re.search(pattern, text))
    return classes == 3 and _shannon_entropy(text) >= 4.0


def _key_words(key: str) -> list[str]:
    """Lower-case words of an env name, header, flag, or camelCase key (``clientSecret`` -> client, secret)."""
    return [word.lower() for word in _KEY_WORD_RE.findall(str(key))]


def _is_query_credential_key(key: str) -> bool:
    """A short key (``key``, ``sig``, ``pass``) that names a credential in a URL query, but not on its own elsewhere."""
    return str(key).strip().lstrip("-").lower() in _QUERY_CREDENTIAL_KEYS


def _is_neutral_value(value: str) -> bool:
    """A boolean, an auth mode, or a path: never secret material by itself."""
    text = value.strip()
    return text.lower() in _NEUTRAL_CREDENTIAL_VALUES or _looks_like_path(text)


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
        return bool(_FILE_EXTENSION_RE.search(segment)) or not looks_like_secret_material(segment)
    if len(segments) > 1 and _BASE64_TEXT_RE.match(rest) and len(rest) >= 32:
        classes = sum(1 for pattern in (r"[a-z]", r"[A-Z]", r"[0-9]") if re.search(pattern, rest))
        return not (classes == 3 and _shannon_entropy(rest) >= 4.5)
    return True


def _shannon_entropy(text: str) -> float:
    counts: dict[str, int] = {}
    for char in text:
        counts[char] = counts.get(char, 0) + 1
    return -sum(count / len(text) * math.log2(count / len(text)) for count in counts.values())


# Where the authority of a URL's raw text ends (a backslash does not end it there).
_RAW_AUTHORITY_END_RE = re.compile(r"[/?#]")


@dataclass(frozen=True)
class UrlCredentials:
    """Where the text of one URL carries credentials; false when it carries none."""

    # The userinfo ('user:password@') carries one.
    userinfo: bool = False
    # Query parameters that carry one, in the order they appear.
    query_keys: tuple[str, ...] = ()
    # Parameters in the fragment ('#access_token=...', '#/cb?token=...') that carry one.
    fragment_keys: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.userinfo or bool(self.query_keys) or bool(self.fragment_keys)


# Which userinfo counts as a credential; see ``url_credentials``.
UserinfoRule = Literal["any", "literal", "secret"]


def url_credentials(url: str, *, userinfo_rule: UserinfoRule) -> UrlCredentials:
    """Credentials written into the text of ``url``: in its userinfo, query parameters, and fragment parameters.

    MCP server URLs, HTTP hook URLs, and URLs inside hook commands all go
    through this one rule; only the userinfo rule differs. The URL is read as
    raw text, so a malformed port or bracket cannot hide a credential: the
    authority runs from ``//`` to the next ``/``, ``?``, or ``#``, and the
    userinfo through its last ``@``.

    * Userinfo, by ``userinfo_rule``:

      - ``"any"``: every user name or password counts, even a reference. An
        HTTP hook's client sends the userinfo as Basic auth with every request.
      - ``"literal"``: a user name or password counts unless it is a reference
        (:func:`is_env_reference`; MCP server URLs, where references are allowed).
      - ``"secret"``: only a literal password counts (a ``${PW:-default}``
        ships its default), or a user name shaped like a token or a long random
        value (a URL inside a command line, and text that only urllib reads as
        userinfo), so ``https://x-access-token:${GITHUB_TOKEN}@github.com/...``
        and the ``169.254.169.254\\`` of ``https://169.254.169.254\\@host/`` do
        not.
    * Query: a parameter counts when its name is shaped like a secret (a bare
      ``?ghp_...``) or a value of it is an inline credential under that name
      (:func:`looks_like_inline_secret`, where ``key=`` and ``sig=`` name one):
      a value shaped like a secret under any name (``q=sk-...``), or a literal
      under a credential name that is not a setting (``api_key=literal``, not
      ``token=true``). References never count. The query ends at the fragment;
      the fragment's parameters (after its last ``?``, as in
      ``#/cb?access_token=...``) are read the same way, as ``fragment_keys``.
    """
    user, password = _split_userinfo(_split_raw_userinfo(url)[0])
    before_fragment, _hash, fragment = url.partition("#")
    return UrlCredentials(
        userinfo=_userinfo_carries_credential(user, password, rule=userinfo_rule),
        query_keys=_credential_parameters(before_fragment.partition("?")[2]),
        fragment_keys=_credential_parameters(fragment.rpartition("?")[2]),
    )


def _credential_parameters(text: str) -> tuple[str, ...]:
    """The names of the parameters in ``text`` (``name=value`` or a bare ``name``) that carry a credential, in order."""
    return tuple(
        dict.fromkeys(
            key
            for key, values in parse_qs(text, keep_blank_values=True).items()
            if has_secret_shape(key) or any(looks_like_inline_secret(key, value, query=True) for value in values)
        )
    )


def _split_raw_userinfo(url: str) -> tuple[str, str]:
    """``(userinfo, the URL without it)``, read from the raw text as :func:`url_credentials` reads it.

    The authority runs from ``//`` to the next ``/``, ``?``, or ``#`` (a
    backslash does not end it), and the userinfo through its last ``@``.
    """
    prefix, slashes, rest = url.partition("//")
    authority = _RAW_AUTHORITY_END_RE.split(rest, maxsplit=1)[0]
    userinfo, at, _host = authority.rpartition("@")
    if not at:
        return "", url
    return userinfo, f"{prefix}{slashes}{rest[len(userinfo) + 1 :]}"


def _split_userinfo(userinfo: str) -> tuple[str, str]:
    """``(user, password)``: the userinfo split at its first ``:`` outside a ``${...}`` reference."""
    index = 0
    while True:
        colon = userinfo.find(":", index)
        if colon == -1:
            return userinfo, ""
        opening = userinfo.rfind("${", index, colon)
        closing = userinfo.find("}", colon) if opening != -1 and userinfo.find("}", opening, colon) == -1 else -1
        if closing == -1:
            return userinfo[:colon], userinfo[colon + 1 :]
        index = closing + 1  # the ':' of '${USER:-}' or '${env:USER}' is inside the reference


def without_userinfo(url: str) -> str:
    """``url`` without the userinfo of its raw text, which :func:`url_credentials` checks for credentials.

    A WHATWG client can read part of that userinfo as the host: it reads a
    backslash as ``/``, so ``https://token\\@example.com/`` is host ``token``.
    """
    return _split_raw_userinfo(url)[1]


def _userinfo_carries_credential(user: str, password: str, *, rule: UserinfoRule) -> bool:
    if rule == "any":
        return bool(user or password)
    if rule == "literal":
        return any(part and not is_env_reference(part) for part in (user, password))
    # Only the text the plugin ships counts: what is left of the password once its references are unset.
    literal_password = bool(env_defaults_text(unquote(password)).strip())
    # A backslash before '@' is part of the raw userinfo, never of a token: WHATWG clients read it as '/'.
    name = unquote(user).strip("\\")
    return literal_password or has_secret_shape(name) or looks_like_random_secret("user", name)


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


def url_scheme(text: str) -> str | None:
    """The lower-case scheme ``text`` starts with (``https`` for ``HTTPS://h/``), or ``None`` when it has none."""
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
    base_scheme = url_scheme(base_url) if base_url is not None else None
    scheme = url_scheme(text)
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


def _path_start(text: str, scheme: str | None) -> int:
    """Index where the path, query, or fragment of a special-scheme URL starts, after ``scheme:`` and its authority."""
    start = len(scheme) + 1 if scheme else 0
    while start < len(text) and text[start] in "/\\":
        start += 1
    end = _AUTHORITY_END_RE.search(text, start)
    return end.start() if end else len(text)


def url_ambiguities(url: str, *, percent_in_host: bool = False) -> list[str]:
    """Why urllib and a WHATWG client (Node, MCP SDKs) could read ``url`` differently, or as different text.

    Flags whitespace or control characters inside the URL (only leading and
    trailing ASCII whitespace is allowed, which both readings strip, and a
    plain space after the host, which clients percent-encode; a NUL, U+2028,
    or no-break space at either end still counts), and invisible
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
    inner_scheme = url_scheme(inner)
    path_start = _path_start(inner, inner_scheme) if inner_scheme in _WHATWG_SPECIAL_SCHEMES else len(inner)
    if any(
        unicodedata.category(char) == "Cc" or (char.isspace() and (index < path_start or char != " "))
        for index, char in enumerate(inner)
    ):
        problems.append("whitespace or a control character")
    if any(unicodedata.category(char) == "Cf" for char in url):
        problems.append("an invisible format character (such as a zero-width space)")
    trimmed = _TAB_OR_NEWLINE_RE.sub("", text.strip(_C0_CONTROL_OR_SPACE))
    scheme = url_scheme(trimmed)
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
    if not cut and len(shown) <= limit:
        return shown
    if limit <= len(_TRUNCATED):
        return shown[:limit]
    return shown[: limit - len(_TRUNCATED)] + _TRUNCATED


# redact_secrets reads whole plugin lines, so these patterns read each run of text once: a pattern tried at every
# word of a long run, reading to the end of the run each time, takes quadratic time (100 KB of '-ab' took 30 s).
#
# A URL's userinfo anywhere in a text: inside '${VAR:-https://user:pw@host}', after another word, or after a
# scheme-less '//'. Inside other text only an authority introduced by two slashes counts, so 'mcp:1.2.3@sha256:...'
# or 'npm:pkg@1.2.3' is left alone. A match starts at the first slash of a run; the scheme before it stays as written.
_EMBEDDED_USERINFO_RE = re.compile(r"(?<![/\\])([/\\]{2,}+)[^\s/\\?#@{}'\"<>]*+@")
_MAX_USERINFO_PASSES = 4
# Where a URL starts inside other text: a scheme, ':', and two or more slashes. The scheme is a run of scheme
# characters ('git+https') with a letter that starts a word (_SCHEME_START_RE), so 'x-https://' has one and
# 'x_https://' does not.
_EMBEDDED_URL_START_RE = re.compile(r"(?i)(?<![a-z0-9+.\-])([a-z0-9+.\-]++):[/\\]{2,}+")
_SCHEME_START_RE = re.compile(r"(?i)\b[a-z]")
# What ends the authority and path of a URL inside other text and, but for '?', its query: a blank, '?', '#', a
# quote, '<', '>', or a brace. A '${...}' reference is part of the URL whatever it holds, up to the next '}'; with
# no '}' after it, its '{' ends the URL.
_URL_TEXT_BREAK_RE = re.compile(r"""[\s?#'"<>{}]|\$\{""")
# Values that read as credentials wherever they appear in an echoed command or spec: 'Bearer <token>', the
# value after a credential-named flag ('--api-key X', '--token=X'), and a credential-named assignment. A name is
# read to its end once, from its first credential word: a later word reaches the same end.
_ECHOED_AUTH_VALUE_RE = re.compile(r"(?i)\b(bearer|basic|token)(\s+)[A-Za-z0-9+/._=~-]{8,}")
_ECHOED_CREDENTIAL_FLAG_RE = re.compile(
    r"(?i)((?<![\w-])--?(?>[\w-]*?(?:token|secret|passw(?:or)?d|api[-_]?key|access[-_]?key|private[-_]?key|auth))"
    r"[\w-]*+(?:=|\s+))(?!\$)[^\s'\"|;&]+"
)
_ECHOED_CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?>[A-Z0-9_]*?(?:TOKEN|SECRET|PASSW(?:OR)?D|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY))[A-Z0-9_]*+=)"
    r"(?!\$)[^\s'\"|;&]+"
)
# The same names with a quoted value, the way shell, Python, YAML, and JSON lines write one: 'KEY="..."',
# "API_KEY = '...'", "os.environ['API_KEY']='...'", 'api_token: "..."', '"password": "..."'. The value runs to the
# matching quote on the same line.
_ECHOED_QUOTED_CREDENTIAL_RE = re.compile(
    r"(?i)(\b(?>[A-Z0-9_]*?(?:TOKEN|SECRET|PASSW(?:OR)?D|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY))[A-Z0-9_]*+"
    r"(?:['\"]\]?)?[ \t]*+[=:][ \t]*+(['\"]))(?!\$)[^'\"\r\n]*+(?=\2)"
)
# A WHATWG client reads 'https:user:pw@host' (a special scheme with no slashes) as user information too. The user
# information stops at the next such scheme, so each scheme is read once ('https:' repeated stays linear).
_SLASHLESS_USERINFO_RE = re.compile(
    r"(?i)(?<![a-z0-9+.\-])((?:https?|wss?|ftp):)(?![/\\])"
    r"(?:(?!(?:https?|wss?|ftp):)[^\s/\\?#@{}'\"<>])*+@"
)
# 'user:password@host/path' with no scheme, starting a word (a container image with registry credentials). The
# '/' after the host keeps 'npm:pkg@1.2.3' and 'mcp:1.2.3@sha256:...' readable; '@scope/pkg' has no user name.
# Each part stops where the next could start ('=' for a user name, ':' for a password), so the text is read in
# linear time.
_SCHEMELESS_USERINFO_RE = re.compile(r"""(?<![^\s'"=])([^\s/@:'"=]++:)[^\s/@:'"]++@(?=[^\s/@'"]++/)""")


def redact_secrets(text: str) -> str:
    """``text`` (a command token, spec, or line, or a quoted code snippet) with its credentials removed.

    URL userinfo (also in ``https:user:pw@host``) and queries are dropped (a
    fragment such as a git ref stays), the password of a scheme-less
    ``user:pw@host/path`` is redacted, and known secret shapes (``ghp_...``,
    ``sk-...``, a JWT), ``Bearer <token>``, the value after a credential-named
    flag, and a credential-named assignment, quoted or not, become
    ``<redacted>``. The rest stays as written, line breaks included, and the
    text is not bounded; it is read in linear time, so a long line costs
    little. Bound it (:func:`report_text`) where a report field shows it.
    """
    for _pass in range(_MAX_USERINFO_PASSES):
        stripped = _EMBEDDED_USERINFO_RE.sub(r"\1", text)
        if stripped == text:
            break
        text = stripped
    text = _SLASHLESS_USERINFO_RE.sub(r"\1", text)
    text = _SCHEMELESS_USERINFO_RE.sub(r"\1<redacted>@", text)
    text = _without_url_queries(text)
    text = _SECRET_VALUE_RE.sub("<redacted>", text)
    text = _ECHOED_AUTH_VALUE_RE.sub(r"\1\2<redacted>", text)
    text = _ECHOED_CREDENTIAL_FLAG_RE.sub(r"\1<redacted>", text)
    text = _ECHOED_QUOTED_CREDENTIAL_RE.sub(r"\1<redacted>", text)
    return _ECHOED_CREDENTIAL_ASSIGNMENT_RE.sub(r"\1<redacted>", text)


def _without_url_queries(text: str) -> str:
    """``text`` without the query of each URL in it: ``scheme://host/path?query`` keeps ``scheme://host/path``.

    A URL's path ends at its first break (``_URL_TEXT_BREAK_RE``). When that
    break is ``?``, the query runs from it to the next break that is not ``?``.
    URLs are read from left to right, and a URL inside the query of an earlier
    one goes with that query. Where a path or a query that reaches each break
    ends is worked out once, from the last break back, so URLs that share a
    path (``a://`` repeated) do not each read it to its end, as the regular
    expression this replaces did.
    """
    urls = [
        url for url in _EMBEDDED_URL_START_RE.finditer(text) if _SCHEME_START_RE.search(text, url.start(), url.end(1))
    ]
    if not urls:
        return text
    breaks = [found.start() for found in _URL_TEXT_BREAK_RE.finditer(text)]
    # Where a path, and a query, that reaches breaks[index] ends.
    path_ends = [len(text)] * (len(breaks) + 1)
    query_ends = [len(text)] * (len(breaks) + 1)
    closing = None  # the index of the next '}'
    for index in reversed(range(len(breaks))):
        start = breaks[index]
        char = text[start]
        if char == "}":
            closing = index
        if char != "$":
            path_ends[index] = start
            query_ends[index] = query_ends[index + 1] if char == "?" else start
        elif closing is not None:
            # A '${' reference ends at the next '}', and the URL goes on after it.
            path_ends[index], query_ends[index] = path_ends[closing + 1], query_ends[closing + 1]
        else:
            path_ends[index] = query_ends[index] = start + 1  # no '}' after it: its '{' ends the URL
    kept: list[str] = []
    resume = 0
    for url in urls:
        if url.start() < resume:
            continue  # inside the query of an earlier URL
        index = bisect.bisect_left(breaks, url.end())
        query = path_ends[index]
        if text.startswith("?", query):
            kept.append(text[resume:query])
            resume = query_ends[index]
    kept.append(text[resume:])
    return "".join(kept)


def report_value(value: str, limit: int = MAX_REPORT_CHARS) -> str:
    """A value from plugin config (a command argument, a package spec) for reports.

    It is withheld whole as ``<value withheld>`` when it is shaped like a
    secret (:func:`has_secret_shape`), and otherwise shown as :func:`report_text`,
    which also removes URL userinfo and redacts ``key=value`` credentials.
    """
    return "<value withheld>" if has_secret_shape(value) else report_text(value, limit)


def safe_url(url: str) -> str:
    """A URL for reports: no userinfo, query, or fragment; bounded (also for a URL that does not parse).

    An http(s), ws(s), or ftp URL is shown the way a WHATWG client (Node, the MCP
    SDKs) reads it, so the report names the host a client would actually contact.
    The userinfo of the raw text is removed first (:func:`without_userinfo`), so
    text the URL holds as its userinfo is never shown, even where a backslash
    makes a client read it as the host. A token in the path is redacted like any
    other report text (:func:`report_text`).
    """
    text = without_userinfo(url)
    try:
        parsed = urlparse(whatwg_url(text))
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
    except ValueError:
        return _unparsed_url(text)
    if not parsed.scheme or not host:
        return _unparsed_url(text)
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
