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
import re
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import parse_qs, unquote, urlparse, urlunparse

import idna

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
# Interpreters invoked with an inline program string execute arbitrary code.
_SHELL_INTERPRETERS: frozenset[str] = frozenset({"sh", "bash", "zsh", "dash", "ksh", "fish"})
# Floating / non-pinned version markers (supply-chain drift risk).
_FLOATING_MARKERS: tuple[str, ...] = ("@latest", "@main", "@master", "@head", "@next", "@canary", ":latest", ":main")

# Command flags that disable TLS/cert verification.
_INSECURE_TLS_FLAGS: frozenset[str] = frozenset(
    {"--insecure", "-k", "--no-check-certificate", "--tls-no-verify", "--ssl-no-verify", "--no-verify-tls"}
)

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
# Known inline-secret value shapes.
_SECRET_VALUE_RE = re.compile(
    r"(sk-[A-Za-z0-9]{16,}"
    r"|ghp_[A-Za-z0-9]{20,}"
    r"|glpat-[A-Za-z0-9_-]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|nvapi-[A-Za-z0-9_-]{16,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})"
)


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
    """True when *value* is an ``$VAR`` / ``${VAR}`` env reference (not a literal)."""
    return bool(_ENV_REF_RE.match(value.strip()))


def _looks_like_inline_secret(key: str, value: str) -> bool:
    """True when an env/header value is an inline credential rather than a reference."""
    v = value.strip()
    if not v or _is_env_reference(v):
        return False
    if _SECRET_VALUE_RE.search(v):
        return True
    # An inline HTTP auth-scheme credential ("Bearer <token>" / "Basic <base64>"),
    # independent of the key name -- covers Authorization-style headers.
    if _INLINE_AUTH_SCHEME_RE.match(v):
        return True
    # A credential-named key whose value is a non-empty, non-reference literal.
    return bool(_SECRET_KEY_RE.search(str(key)))


def _credential_flag_name(token: str) -> str | None:
    """Return the flag name when *token* is a credential-bearing option flag.

    Handles ``--api-key`` / ``--api-key=VALUE`` (and short ``-x`` / ``-x=VALUE``)
    forms. The flag name (leading dashes stripped) is matched against the same
    credential vocabulary used for env keys (:data:`_SECRET_KEY_RE`).
    """
    if not token.startswith("-"):
        return None
    flag = token.lstrip("-").split("=", 1)[0]
    return flag if flag and _SECRET_KEY_RE.search(flag) else None


def redacted_url(url: str) -> str:
    """URL for finding messages: scheme, host, port, and path only.

    Userinfo, parameters, query, and fragment are dropped so an inline credential
    is never echoed into reports or CI logs.
    """
    parsed = urlparse(url.strip())
    return urlunparse((parsed.scheme, parsed.netloc.rpartition("@")[2], parsed.path, "", "", ""))


def _check_url_inline_secrets(name: str, url: str, parsed: Any, file_path: str, findings: list[Finding]) -> None:
    """Flag inline credentials embedded in a URL's userinfo or query string."""
    try:
        username, password = parsed.username, parsed.password
    except ValueError:  # malformed netloc / port
        username = password = None
    if (password and not _is_env_reference(password)) or (username and not _is_env_reference(username)):
        findings.append(
            _finding(
                Severity.CRITICAL,
                "mcp_url_inline_secret",
                f"url embeds inline userinfo credentials: {redacted_url(url)!r} (userinfo withheld); only "
                "${ENV} references are allowed",
                file_path,
                'Remove user:password@ from the URL; pass credentials by reference (e.g. header "${MY_TOKEN}").',
                name=name,
            )
        )
    for key, values in parse_qs(parsed.query, keep_blank_values=True).items():
        if not _SECRET_KEY_RE.search(key):
            continue
        if any(v and not _is_env_reference(v) for v in values):
            findings.append(
                _finding(
                    Severity.CRITICAL,
                    "mcp_url_inline_secret",
                    f"url query parameter {key!r} carries an inline credential; only ${{ENV}} references are allowed",
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
    for token in tokens:
        if _SHELL_METACHAR_RE.search(token):
            findings.append(
                _finding(
                    Severity.CRITICAL,
                    "mcp_command_shell_metacharacters",
                    f"command token contains shell metacharacters: {token!r}",
                    file_path,
                    "Remove shell operators (; | & ` $() < >). MCP commands run argv-style, not via a shell.",
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
        low = token.lower()
        if any(marker in low for marker in _FLOATING_MARKERS):
            findings.append(
                _finding(
                    Severity.HIGH,
                    "mcp_command_floating_version",
                    f"command token uses a floating (unpinned) version: {token!r}",
                    file_path,
                    "Pin the referenced package/image to an exact version, not latest/main.",
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
            if value and not _is_env_reference(value):
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
                    f"command argument contains an inline credential: {token!r}",
                    file_path,
                    'Pass the secret by reference (e.g. "${MY_TOKEN}"); never inline a raw credential in args.',
                    name=name,
                )
            )

    # Shell interpreter invoked with an inline program string (`sh -c "..."`).
    base = command.strip().split("/")[-1].split("\\")[-1].lower()
    if base in _SHELL_INTERPRETERS and any(str(a).strip() == "-c" for a in (args or [])):
        findings.append(
            _finding(
                Severity.CRITICAL,
                "mcp_command_dangerous_form",
                f"command invokes a shell interpreter with '-c' ({command!r}); this executes an arbitrary program string",
                file_path,
                "Invoke the server binary directly instead of wrapping it in a shell '-c' string.",
                name=name,
            )
        )


def _validate_url(
    name: str,
    config: dict[str, Any],
    file_path: str,
    findings: list[Finding],
    allowed_private_hosts: Iterable[str] = (),
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

    parsed = urlparse(url.strip())
    scheme = (parsed.scheme or "").lower()
    shown = redacted_url(url)  # messages never echo userinfo or query credentials
    # Inline credentials in userinfo/query are persisted verbatim; check them
    # independent of the scheme (secure https URLs are the common case).
    _check_url_inline_secrets(name, url, parsed, file_path, findings)
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
                    f"url has a malformed authority/port: {shown!r}",
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
                    f"url uses scheme {scheme!r} but has no host to connect to: {shown!r}",
                    file_path,
                    "Provide a full endpoint with a hostname, e.g. https://host[:port]/path.",
                    name=name,
                )
            )
        _validate_endpoint(name, url, host, file_path, findings, allowed_private_hosts)
        return
    if scheme in _INSECURE_URL_SCHEMES:
        # Plaintext endpoints are blocked below; still report where they point.
        try:
            insecure_host = parsed.hostname
        except ValueError:
            insecure_host = None
        _validate_endpoint(name, url, insecure_host, file_path, findings, allowed_private_hosts)
    if scheme in _DANGEROUS_URL_SCHEMES or scheme == "":
        findings.append(
            _finding(
                Severity.CRITICAL,
                "mcp_url_dangerous_scheme",
                f"url uses a dangerous/invalid scheme {scheme or '(none)'!r}: {shown!r}",
                file_path,
                "Use a secure https:// or wss:// endpoint; file/data/javascript/ftp schemes are not permitted.",
                name=name,
            )
        )
    elif scheme in _INSECURE_URL_SCHEMES:
        findings.append(
            _finding(
                Severity.HIGH,
                "mcp_url_insecure_scheme",
                f"url uses an insecure plaintext scheme {scheme!r}: {shown!r}",
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
                f"url scheme {scheme!r} is not an allowed MCP scheme: {shown!r}",
                file_path,
                f"Use one of the allowed secure schemes: {', '.join(sorted(ALLOWED_MCP_URL_SCHEMES))}.",
                name=name,
            )
        )


def _validate_env_and_headers(name: str, config: dict[str, Any], file_path: str, findings: list[Finding]) -> None:
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
        # NON-BLOCKING advisory: the evaluation runtime applies command+args (stdio)
        # and url (http/sse) only -- Harbor's per-MCP-server config has no env/headers
        # field and no agent adapter emits them, so this block will not reach the
        # launched MCP server (use task-level environment / CI credential injection
        # instead). The inline-secret / insecure-TLS checks below still run, so a raw
        # credential declared here is still caught and blocks.
        findings.append(
            _finding(
                Severity.LOW,
                "mcp_field_ignored",
                f"'{section}' is not applied by the evaluation runtime and will be ignored; "
                "a Tier 3 run of this server is reported INCOMPLETE",
                file_path,
                f"Remove '{section}' or rely on task-level environment / CI credential injection; "
                "the runtime applies command+args (stdio) and url (http/sse) only.",
                name=name,
            )
        )
        for key, value in block.items():
            if not isinstance(value, str):
                continue
            if _is_insecure_tls_env(key, value):
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
            if _looks_like_inline_secret(key, value):
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
_PLUGIN_PATH_REFS: tuple[str, ...] = ("${CLAUDE_PLUGIN_ROOT}", "${CLAUDE_PLUGIN_DATA}", "${CLAUDE_PROJECT_DIR}")
_LOCAL_SPEC_PREFIXES: tuple[str, ...] = (".", "/", "~", "file:", *_PLUGIN_PATH_REFS)
_REMOTE_SPEC_PREFIXES: tuple[str, ...] = ("git+", "git:", "github:", "gitlab:", "bitbucket:", "http://", "https://")

# Value-taking flags per package runner, so a flag's value is never mistaken for
# the package spec. Unknown flags are treated as boolean.
_NPX_VALUE_FLAGS = frozenset(
    {"-p", "--package", "-c", "--call", "--registry", "--cache", "--userconfig", "--prefix", "-w", "--workspace"}
)
_DLX_VALUE_FLAGS = frozenset({"-p", "--package", "--registry", "--allow-build"})
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


@dataclass(frozen=True)
class McpPinning:
    """Static supply-chain pinning classification of one MCP declaration."""

    status: PinStatus
    detail: str

    @property
    def pinned(self) -> bool | None:
        """``True``/``False`` for package runners; ``None`` when not applicable."""
        if self.status == "not_applicable":
            return None
        return self.status == "pinned"


def _command_basename(command: str) -> str:
    base = command.strip().replace("\\", "/").rsplit("/", 1)[-1].lower()
    for suffix in (".exe", ".cmd", ".bat", ".ps1"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


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


def _classify_npm_spec(spec: str) -> McpPinning:
    """Classify an npm package spec (``pkg``, ``@scope/pkg@1.2.3``, git/URL, local path)."""
    if _is_local_spec(spec):
        return McpPinning("not_applicable", f"local package path {spec!r}")
    if spec.startswith(_REMOTE_SPEC_PREFIXES) or (not spec.startswith("@") and "/" in spec):
        if _GIT_SHA_RE.search(spec):
            return McpPinning("pinned", f"git/URL spec pinned to a commit: {spec!r}")
        return McpPinning("unpinned", f"git/URL/GitHub spec without a commit SHA: {spec!r}")
    at = spec.find("@", 1) if spec.startswith("@") else spec.find("@")
    if at <= 0:
        return McpPinning("unpinned", f"package {spec!r} has no version (resolves to the latest release)")
    version = spec[at + 1 :]
    if _EXACT_SEMVER_RE.match(version):
        return McpPinning("pinned", f"exact version {spec!r}")
    return McpPinning("unpinned", f"package {spec!r} uses a version range or dist-tag, not an exact version")


def _classify_python_spec(spec: str) -> McpPinning:
    """Classify a PyPI requirement spec as used by ``uvx`` / ``pipx run``."""
    if _is_local_spec(spec):
        return McpPinning("not_applicable", f"local package path {spec!r}")
    if spec.startswith(_REMOTE_SPEC_PREFIXES) or "@ git+" in spec or "@git+" in spec:
        if _GIT_SHA_RE.search(spec) or "#sha256=" in spec:
            return McpPinning("pinned", f"git/URL spec pinned to a commit or hash: {spec!r}")
        return McpPinning("unpinned", f"git/URL spec without a commit SHA or hash: {spec!r}")
    if _PEP440_EXACT_RE.match(spec):
        return McpPinning("pinned", f"exact version {spec!r}")
    name, sep, version = spec.partition("@")
    if sep and name and _EXACT_SEMVER_RE.match(version.strip()):
        return McpPinning("pinned", f"exact version {spec!r}")
    if any(marker in spec for marker in ("<", ">", "~", "!", "*", ",", "=", "@")):
        return McpPinning("unpinned", f"requirement {spec!r} is a range or tag, not an exact '==' version")
    return McpPinning("unpinned", f"package {spec!r} has no version (resolves to the latest release)")


def _classify_image(image: str) -> McpPinning:
    """Classify a container image reference."""
    if _DOCKER_DIGEST_RE.search(image):
        return McpPinning("pinned", f"image pinned by digest {image!r}")
    if "${" in image or image.startswith("$"):
        return McpPinning("unpinned", f"image {image!r} is taken from an environment reference")
    last = image.rsplit("/", 1)[-1]
    tag = last.split(":", 1)[1] if ":" in last else None
    if tag is None:
        return McpPinning("unpinned", f"image {image!r} has no tag or digest (implicit ':latest')")
    if tag.lower() == "latest":
        return McpPinning("unpinned", f"image {image!r} uses the mutable 'latest' tag")
    if _VERSION_TAG_RE.match(tag):
        return McpPinning("pinned", f"image {image!r} uses a version tag (tags are mutable; a digest is stronger)")
    return McpPinning("unpinned", f"image {image!r} uses the non-version tag {tag!r}")


def _classify_spec_list(specs: list[str], classify: Any) -> McpPinning:
    results = [classify(spec) for spec in specs]
    unpinned = [result for result in results if result.status == "unpinned"]
    if unpinned:
        return unpinned[0]
    pinned = [result for result in results if result.status == "pinned"]
    if pinned:
        return pinned[0]
    return results[0]


def classify_mcp_pinning(config: Any) -> McpPinning:
    """Classify whether one MCP declaration runs an exactly-pinned package.

    Package runners (``npx``, ``bunx``, ``pnpm dlx``, ``yarn dlx``, ``npm exec``,
    ``uvx``, ``uv tool run``, ``pipx run``, ``deno run`` of a registry spec, and
    ``docker|podman run``) are ``pinned`` only with an exact version
    (``pkg@1.2.3``, ``pkg==1.2.3``, ``--from pkg==1.2.3``, ``image:1.2.3``,
    ``image@sha256:...``); otherwise they are ``unpinned``. Local interpreters and
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
    command_parts = command.split()
    if len(command_parts) > 1:
        # A whole command line in 'command' ("npx -y pkg"): classify it argv-style.
        command, args = command_parts[0], [*command_parts[1:], *args]
    base = _command_basename(command)

    if base in {"npx", "bunx", "pnpx"} or (base in {"pnpm", "yarn"} and args[:1] == ["dlx"]):
        rest = args[1:] if base in {"pnpm", "yarn"} else args
        value_flags = _NPX_VALUE_FLAGS if base == "npx" else _DLX_VALUE_FLAGS
        runner = f"{base} dlx" if base in {"pnpm", "yarn"} else base
        return _classify_npm_runner(runner, rest, value_flags)
    if base == "npm" and args[:1] in (["exec"], ["x"]):
        return _classify_npm_runner("npm exec", args[1:], _NPX_VALUE_FLAGS)
    if base == "uvx" or (base == "uv" and args[:2] in (["tool", "run"], ["tool", "x"])):
        rest = args if base == "uvx" else args[2:]
        from_values = _flag_values(rest, ("--from",))
        spec = from_values[0] if from_values else _first_positional(rest, _UVX_VALUE_FLAGS)
        if spec is None:
            return McpPinning("unpinned", "uvx invocation without a package spec")
        return _prefixed(f"{base if base == 'uvx' else 'uv tool run'}: ", _classify_python_spec(spec))
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
    if base in _CONTAINER_RUNTIMES:
        if args[:1] == ["run"]:
            rest = args[1:]
        elif args[:2] == ["container", "run"]:
            rest = args[2:]
        else:
            return McpPinning("not_applicable", f"{base} invocation is not 'run'")
        image = _first_positional(rest, _DOCKER_VALUE_FLAGS)
        if image is None:
            return McpPinning("unpinned", f"{base} run invocation without an image")
        return _prefixed(f"{base} run: ", _classify_image(image))
    return McpPinning("not_applicable", f"local interpreter, script, or binary ({base!r})")


def _classify_npm_runner(runner: str, tokens: list[str], value_flags: frozenset[str]) -> McpPinning:
    packages = _flag_values(tokens, ("-p", "--package"))
    if packages:
        return _prefixed(f"{runner}: ", _classify_spec_list(packages, _classify_npm_spec))
    spec = _first_positional(tokens, value_flags)
    if spec is None:
        return McpPinning("not_applicable", f"{runner} invocation without a package spec")
    return _prefixed(f"{runner}: ", _classify_npm_spec(spec))


def _prefixed(prefix: str, pin: McpPinning) -> McpPinning:
    return McpPinning(pin.status, f"{prefix}{pin.detail}")


# --------------------------------------------------------------------------- #
# Network-free endpoint policy                                                #
# --------------------------------------------------------------------------- #
EndpointKind = Literal["metadata", "private"]

_METADATA_HOSTNAMES = frozenset({"metadata.google.internal", "metadata"})
_METADATA_ADDRESSES = frozenset(
    {
        ipaddress.ip_address("169.254.169.254"),
        ipaddress.ip_address("fd00:ec2::254"),
        ipaddress.ip_address("100.100.100.200"),
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
    (ipaddress.ip_network("::1/128"), "loopback"),
    (ipaddress.ip_network("::/128"), "unspecified"),
    (ipaddress.ip_network("fc00::/7"), "unique local (fc00::/7)"),
    (ipaddress.ip_network("fe80::/10"), "link-local"),
    (ipaddress.ip_network("fec0::/10"), "site-local (deprecated)"),
)
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
    encoded = encoded or name_encoded
    found = _address_class(address)
    if found is None:
        inner = _embedded_ipv4(address)
        if inner is not None:
            found = _address_class(inner)
            encoded = True
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


def _validate_endpoint(
    name: str,
    url: str,
    host: str | None,
    file_path: str,
    findings: list[Finding],
    allowed_private_hosts: Iterable[str],
) -> None:
    if not host:
        return
    endpoint = classify_endpoint_host(host)
    if endpoint is None:
        return
    encoded = f" (encoded as {host!r})" if endpoint.encoded else ""
    shown = redacted_url(url)  # never echo userinfo or query credentials
    if endpoint.kind == "metadata":
        findings.append(
            _finding(
                Severity.HIGH,
                "mcp_endpoint_metadata",
                f"url targets a {endpoint.reason} endpoint{encoded}: {shown!r}; an MCP client pointed here "
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
            f"url host is a {endpoint.reason} address{encoded}: {shown!r}; the endpoint is not publicly "
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
    "--allow-dangerously-skip-permissions",
    "--yolo",
)
_BYPASS_FLAG_RE = re.compile(
    r"(?<![\w-])(" + "|".join(re.escape(flag) for flag in PERMISSION_BYPASS_FLAGS) + r")(?![\w-])",
    re.IGNORECASE,
)
# Option/value pairs with the same effect: Claude Code's permission mode, Gemini
# CLI's approval mode, and Codex CLI's sandbox (long and short option).
PERMISSION_BYPASS_OPTIONS: tuple[tuple[str, str], ...] = (
    ("--permission-mode", "bypassPermissions"),
    ("--approval-mode", "yolo"),
    ("--sandbox", "danger-full-access"),
    ("-s", "danger-full-access"),
)
# In one string: "--opt value", "--opt=value", or a quoted value.
_BYPASS_OPTION_RES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (
        f"{option} {value}",
        re.compile(rf"(?<![\w-]){re.escape(option)}(?:=|\s+)[\"']?{re.escape(value)}(?![\w-])", re.IGNORECASE),
    )
    for option, value in PERMISSION_BYPASS_OPTIONS
)
# Split across adjacent argv tokens: option -> (value, reported flag).
_BYPASS_OPTION_VALUES: dict[str, tuple[str, str]] = {
    option.lower(): (value.lower(), f"{option} {value}") for option, value in PERMISSION_BYPASS_OPTIONS
}
# Keys whose values are prose, never executed config -- documentation mentions of
# a flag are not flagged.
_DOC_KEYS = frozenset({"description", "title", "summary", "notes", "note", "comment", "comments", "help"})
# Matches the load_bounded_json node limit, so every config it accepts is walked
# completely; a walk that still hits the bound is reported, never silently cut.
_MAX_SCAN_NODES = MAX_STRUCTURED_NODES
_CODE_INJECTION_ENV = frozenset({"LD_PRELOAD", "LD_AUDIT", "DYLD_INSERT_LIBRARIES"})
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
        "env_code_injection",
        "env_traffic_redirect",
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


def _bypass_hits(path: str, node: Any) -> Iterator[tuple[str, str]]:
    """Yield ``(json_path, flag)`` for bypass flags in a string or split across argv tokens."""
    if isinstance(node, str):
        for match in _BYPASS_FLAG_RE.finditer(node):
            yield path, match.group(1).lower()
        for label, pattern in _BYPASS_OPTION_RES:
            if pattern.search(node):
                yield path, label
    elif isinstance(node, list):
        # argv lists carry an option and its value as two tokens: ["--sandbox", "danger-full-access"].
        for index, (option, value) in enumerate(itertools.pairwise(node)):
            if not isinstance(option, str) or not isinstance(value, str):
                continue
            expected = _BYPASS_OPTION_VALUES.get(option.strip().lower())
            if expected is not None and value.strip().strip("\"'").lower() == expected[0]:
                yield f"{path}[{index}]", expected[1]


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
    """Permission-bypass flags anywhere in the entry, dangerous env, and auto-approve keys."""
    _append_override_findings(name, permission_bypass_issues(config), file_path, findings)
    _append_override_findings(name, env_override_issues(config.get("env")), file_path, findings)
    _append_override_findings(name, auto_approve_issues(config), file_path, findings)


def _validate_pinning(name: str, config: dict[str, Any], file_path: str, findings: list[Finding]) -> None:
    pin = classify_mcp_pinning(config)
    if pin.status != "unpinned":
        return
    # A floating marker (@latest, :latest, ...) already raised the blocking
    # mcp_command_floating_version finding for this entry; do not double-report.
    if any(f.check_name == "mcp_command_floating_version" and f.metadata.get("mcp_server") == name for f in findings):
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


def validate_mcp_server_declaration(
    name: Any,
    config: Any,
    file_path: str,
    *,
    allowed_private_hosts: Iterable[str] = (),
) -> list[Finding]:
    """Statically validate one contained ``mcpServers`` entry (``name`` -> config).

    ``allowed_private_hosts`` comes from the validation policy
    (``mcp.allowed_private_hosts``) and suppresses ``mcp_endpoint_private`` for
    intended private hosts; cloud metadata endpoints are never allowlisted.
    """
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
    _validate_env_and_headers(name, config, file_path, findings)
    _validate_overrides(name, config, file_path, findings)

    if has_command:
        _validate_command(name, config, file_path, findings)
        _validate_pinning(name, config, file_path, findings)
    if has_url:
        _validate_url(name, config, file_path, findings, allowed_private_hosts)
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
                    validate_contained_mcp_servers(entry, file_path, allowed_private_hosts=allowed_private_hosts)
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
            validate_mcp_server_declaration(name, config, file_path, allowed_private_hosts=allowed_private_hosts)
        )
    return findings
