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
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from skillevaluator.models.plugin import MCP_NAME_PATTERN
from skillevaluator.models.result import Finding, Severity

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
                f"url embeds inline userinfo credentials: {url!r}; only ${{ENV}} references are allowed",
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


def _validate_url(name: str, config: dict[str, Any], file_path: str, findings: list[Finding]) -> None:
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
                    f"url has a malformed authority/port: {url!r}",
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
                    f"url uses scheme {scheme!r} but has no host to connect to: {url!r}",
                    file_path,
                    "Provide a full endpoint with a hostname, e.g. https://host[:port]/path.",
                    name=name,
                )
            )
        return
    if scheme in _DANGEROUS_URL_SCHEMES or scheme == "":
        findings.append(
            _finding(
                Severity.CRITICAL,
                "mcp_url_dangerous_scheme",
                f"url uses a dangerous/invalid scheme {scheme or '(none)'!r}: {url!r}",
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
                f"url uses an insecure plaintext scheme {scheme!r}: {url!r}",
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
                f"url scheme {scheme!r} is not an allowed MCP scheme: {url!r}",
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


def validate_mcp_server_declaration(name: Any, config: Any, file_path: str) -> list[Finding]:
    """Statically validate one contained ``mcpServers`` entry (``name`` -> config)."""
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

    if has_command:
        _validate_command(name, config, file_path, findings)
    if has_url:
        _validate_url(name, config, file_path, findings)
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


def validate_contained_mcp_servers(mcp_servers: Any, file_path: str) -> list[Finding]:
    """Statically validate a contained ``.claude-plugin/plugin.json`` ``mcpServers`` map.

    Returns a (possibly empty) list of blocking :class:`Finding` objects. An
    absent or empty map yields no findings.
    """
    if mcp_servers is None:
        return []
    if not isinstance(mcp_servers, dict):
        return [
            _finding(
                Severity.HIGH,
                "mcp_servers_not_object",
                "'mcpServers' must be a JSON object mapping server names to their config",
                file_path,
                'Express mcpServers as an object: {"<name>": {"command"|"url"|"provider": ...}}.',
            )
        ]
    findings: list[Finding] = []
    for name, config in mcp_servers.items():
        findings.extend(validate_mcp_server_declaration(name, config, file_path))
    return findings
