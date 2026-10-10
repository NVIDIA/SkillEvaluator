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

import contextlib
import ipaddress
import itertools
import re
import shlex
import unicodedata
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, replace
from typing import Any, Literal
from urllib.parse import unquote, urlparse, urlsplit

import idna

from skillevaluator.constants import (
    PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE,
    PLUGIN_CODEX_MANIFEST_TYPE,
    PLUGIN_CURSOR_MANIFEST_TYPE,
)
from skillevaluator.models.plugin import MCP_NAME_PATTERN
from skillevaluator.models.result import Finding, Severity
from skillevaluator.utils.structured_data import MAX_STRUCTURED_NODES
from skillevaluator.validators.url_policy import (
    MAX_REPORT_CHARS,
    env_defaults_text,
    has_secret_shape,
    is_credential_key,
    looks_like_inline_secret,
    looks_like_random_secret,
    redact_secrets,
    report_text,
    safe_url,
    url_ambiguities,
    url_credentials,
    url_scheme,
    whatwg_url,
    without_userinfo,
)

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
_SHELL_METACHAR_RE = re.compile(r"[;&|`\n\r<>]|\$\(")
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

# Values that switch an environment or command-line setting on.
TRUTHY_VALUES: frozenset[str] = frozenset({"1", "true", "yes", "on"})
# Command flags that disable TLS/cert verification.
_INSECURE_TLS_FLAGS: frozenset[str] = frozenset(
    {"--insecure", "-k", "--no-check-certificate", "--tls-no-verify", "--ssl-no-verify", "--no-verify-tls"}
)

# Environment references in an MCP URL: Claude Code expands '${NAME}' and '${NAME:-default}' (the default when
# NAME is unset), and Cursor documents '${env:NAME}'. Only these braced forms are expanded inside a URL.
_URL_EXPANSION_RE = re.compile(r"\$\{(?P<cursor>env:)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")
# The MCP client that loads a manifest format decides how environment references in its MCP config expand.
McpClient = Literal["claude", "codex", "cursor", "agent_plugins"]
_CLIENT_BY_MANIFEST: dict[str, McpClient] = {
    PLUGIN_CODEX_MANIFEST_TYPE: "codex",
    PLUGIN_CURSOR_MANIFEST_TYPE: "cursor",
    PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE: "agent_plugins",
}
# Codex MCP fields that carry auth or environment the evaluation runtime does not apply (Tier 3 then reports
# the server INCOMPLETE).
CODEX_UNAPPLIED_MCP_FIELDS: tuple[str, ...] = (
    "env_vars",
    "env_http_headers",
    "bearer_token_env_var",
    "http_headers_helper",
    "oauth",
)


def mcp_client_for_manifest(manifest_type: str | None) -> McpClient:
    """The client whose MCP config rules apply to a manifest format (Claude Code by default)."""
    return _CLIENT_BY_MANIFEST.get(manifest_type or "", "claude")


# --------------------------------------------------------------------------- #
# Inline credentials in commands                                              #
# --------------------------------------------------------------------------- #
def _credential_flag_name(token: str) -> str | None:
    """Return the flag name when *token* is a credential-bearing option flag.

    Handles ``--api-key`` / ``--api-key=VALUE`` (and short ``-x`` / ``-x=VALUE``)
    forms. The flag name (leading dashes stripped) is matched against the
    credential vocabulary of env keys and query parameters
    (:func:`~skillevaluator.validators.url_policy.is_credential_key`), so the
    short ``--key`` and ``--pass`` count too; their value must then look like
    key material.
    """
    if not token.startswith("-"):
        return None
    flag = token.lstrip("-").split("=", 1)[0]
    return flag if flag and is_credential_key(flag, query=True) else None


def _inline_credentials(words: list[str]) -> list[tuple[int, str | None]]:
    """``(index, flag)`` for each of ``words`` that holds an inline credential, in order.

    The value of a credential-named flag counts: ``--token=x`` itself, or the
    word after ``--token`` unless it looks like another flag (``--api-key
    --verbose``); ``flag`` names the flag. A setting (``--auth basic``, a path)
    is not a credential, and a short flag such as ``--key`` or ``--pass`` needs
    a value shaped like key material (``--key primary`` is a setting). Any other
    word shaped like a secret or an inline ``Bearer``/``Basic`` credential
    counts with ``flag`` ``None``. ``${ENV}`` references never count.
    """
    found: list[tuple[int, str | None]] = []
    value_index = -1
    for index, word in enumerate(words):
        flag = _credential_flag_name(word)
        if flag is not None:
            if "=" in word:
                value, value_index = word.split("=", 1)[1], index
            elif index + 1 < len(words) and not words[index + 1].startswith("-"):
                value, value_index = words[index + 1], index + 1
            else:
                value, value_index = "", -1
            if value and looks_like_inline_secret(flag, value, query=False):
                found.append((value_index, flag))
        elif index != value_index and has_secret_shape(env_defaults_text(word)):
            found.append((index, None))
    return found


# --------------------------------------------------------------------------- #
# Plugin text in findings                                                     #
# --------------------------------------------------------------------------- #
# Stand-in for a reference with no default, so the URL can be parsed to see what the reference decides.
_ENV_MARKER = "zzenvrefzz"


def _shown(text: str, limit: int = 120) -> str:
    """Plugin text (a command token or line, a package spec, a pin detail) for messages: no credential, bounded.

    :func:`~skillevaluator.validators.url_policy.report_text` bounds the text to
    ``limit`` characters first, removing URL user information and the
    credentials it knows; then :func:`redact_secrets` removes URL queries,
    credential flag values, and the other secret shapes.
    """
    return redact_secrets(report_text(text, limit))


def redacted_url(url: str) -> str:
    """URL for finding messages: scheme, host, port, and path only.

    Userinfo, parameters, query, and fragment are dropped so an inline credential
    is never echoed into reports or CI logs, however the authority is written,
    and also when the URL sits inside an environment reference
    (``${MCP_URL:-https://user:pw@host/mcp}``) or after other text. A URL without
    references is shown as :func:`~skillevaluator.validators.url_policy.safe_url`
    shows it: the way a WHATWG client reads it, bounded.
    """
    text = url.strip()
    if _URL_EXPANSION_RE.search(text):
        return redact_secrets(report_text(_redacted_reference_url(text)))
    return safe_url(text)


def _url_display_mask(url: str) -> list[bool]:
    """For each character of ``url``: whether a finding may show it (scheme, host, port, and path).

    User information (through the last ``@`` of the authority) and everything
    from the first ``?`` or ``#`` on are hidden.
    """
    keep = [True] * len(url)
    scheme = url_scheme(url)
    start = len(scheme) + 1 if scheme else 0
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
    return "".join(shown)


# --------------------------------------------------------------------------- #
# Findings                                                                    #
# --------------------------------------------------------------------------- #
def mcp_finding(
    severity: Severity, check_name: str, message: str, file_path: str, suggestion: str, *, name: str | None = None
) -> Finding:
    """An ``MCP_DECLARATION`` finding; with ``name``, the message starts with ``mcpServers['<name>']: ``."""
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


@dataclass
class _ServerFindings:
    """Where findings about one ``mcpServers`` entry go; each message starts with ``mcpServers['<name>']: ``."""

    name: str | None
    file_path: str
    findings: list[Finding] = field(default_factory=list)

    def report(self, severity: Severity, check_name: str, message: str, suggestion: str) -> None:
        self.findings.append(mcp_finding(severity, check_name, message, self.file_path, suggestion, name=self.name))


# --------------------------------------------------------------------------- #
# URL credentials                                                             #
# --------------------------------------------------------------------------- #
def _client_reading(url: str) -> str:
    """How WHATWG clients read an ambiguous URL, for messages: no userinfo, query, or fragment."""
    try:
        parsed = urlparse(whatwg_url(url))
        host = parsed.hostname
        _ = parsed.port  # property access raises ValueError on a malformed port
    except ValueError:  # e.g. 'https://user:password\@host' is host 'user' with port 'password'
        return "reject it as invalid"
    if not host:
        return "find no host in it"
    if host != _whatwg_hostname(without_userinfo(url)):
        # 'https://token\@host': the host they read is text that the URL holds as its userinfo.
        return "read part of its user information as the host"
    return f"read it as {safe_url(url)!r}"


def _safe_hostname(parsed: Any) -> str | None:
    try:
        return parsed.hostname
    except ValueError:
        return None


def _whatwg_hostname(url: str) -> str | None:
    """The host a WHATWG client reads from ``url``; ``None`` when it finds none or cannot parse the URL."""
    try:
        return urlparse(whatwg_url(url)).hostname
    except ValueError:
        return None


def _check_url_inline_secrets(
    server: _ServerFindings, display: str, label: str, *, text: str, ambiguous: bool = False
) -> bool:
    """Flag credentials written into an MCP URL; return whether its user information carries one.

    The URL ``text`` (as the client builds it) goes through the credential
    rule HTTP hooks and hook commands share
    (:func:`~skillevaluator.validators.url_policy.url_credentials`). The
    userinfo is the one WHATWG clients send (also in ``https:user:password@host``),
    where a literal user name or password counts and a reference does not.
    When the URL is ``ambiguous``, urllib reads another authority: a literal
    password or a token-shaped user name there still ships in the plugin text,
    but other text before a backslash (``169.254.169.254\\@host``) is the host
    WHATWG clients connect to. ``display`` is only shown, redacted.
    """
    readings = [url_credentials(whatwg_url(text), userinfo_rule="literal")]
    if ambiguous:
        readings.append(url_credentials(text.strip(), userinfo_rule="secret"))
    userinfo = any(reading.userinfo for reading in readings)
    if userinfo:
        server.report(
            Severity.CRITICAL,
            "mcp_url_inline_secret",
            f"{label} embeds inline userinfo credentials: {redacted_url(display)!r} (userinfo withheld); only "
            "${ENV} references are allowed",
            'Remove user:password@ from the URL; pass credentials by reference (e.g. header "${MY_TOKEN}").',
        )
    for part, keys in (
        ("query", [key for reading in readings for key in reading.query_keys]),
        ("fragment", [key for reading in readings for key in reading.fragment_keys]),
    ):
        for key in dict.fromkeys(keys):
            shown = "<redacted>" if has_secret_shape(key) else key[:64]
            server.report(
                Severity.CRITICAL,
                "mcp_url_inline_secret",
                f"{label} {part} parameter {shown!r} carries an inline credential; only ${{ENV}} references are "
                "allowed",
                "Do not put credentials in the URL query or fragment; reference a secret handle/env var instead.",
            )
    return userinfo


# --------------------------------------------------------------------------- #
# Commands                                                                    #
# --------------------------------------------------------------------------- #
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
        return v in TRUTHY_VALUES
    return False


# Windows command-line wrappers that parse their arguments as one command line, by their switches.
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


def validate_mcp_command(name: str, config: dict[str, Any], file_path: str, findings: list[Finding]) -> None:
    """Append the stdio command findings for one server's ``command`` and ``args`` to ``findings``.

    Shell metacharacters where a shell would read them, a shell's inline
    program (``-c``), inline credentials, and flags that disable TLS. Messages
    start with ``mcpServers['<name>']: ``. Floating versions are a pinning
    finding (:func:`validate_mcp_pinning`).
    """
    _check_command(_ServerFindings(name, file_path, findings), config)


def validate_mcp_pinning(name: str, config: dict[str, Any], file_path: str, findings: list[Finding]) -> None:
    """Append the pinning finding for one server to ``findings``, when its package runner is not pinned.

    A HIGH ``mcp_command_floating_version`` for a moving tag (``@latest``,
    ``:main``) where the runner reads the version, else a MEDIUM
    ``mcp_unpinned_package`` (see :func:`classify_mcp_pinning`).
    """
    _check_pinning(_ServerFindings(name, file_path, findings), config)


def _check_command(server: _ServerFindings, config: dict[str, Any]) -> None:
    command = config.get("command")
    if not isinstance(command, str) or not command.strip():
        server.report(
            Severity.HIGH,
            "mcp_command_empty",
            "runnable MCP 'command' must be a non-empty string",
            "Set 'command' to the server executable (argv-style, no shell string).",
        )
        return

    args = config.get("args")
    if args is not None and not isinstance(args, list):
        server.report(
            Severity.HIGH,
            "mcp_args_not_list",
            "runnable MCP 'args' must be a list of strings",
            "Express command arguments as a JSON array of strings.",
        )

    # Inline credentials: a credential-named flag (--api-key, --token, --password, ...)
    # must reference an env var, never a raw literal, and any argument shaped like a
    # secret or an inline "Bearer/Basic <token>" is flagged whatever the flag name. A
    # command line written in 'command' is read the same way. ${ENV} references are
    # always allowed. The tokens that hold one are never shown in any finding.
    arg_list = [str(a) for a in args] if isinstance(args, list) else []
    command_credentials = _inline_credentials(_split_command_line([command]))
    arg_credentials = _inline_credentials(arg_list)
    tokens = [command, *arg_list]
    # Positions in tokens: the command, then each argument.
    withheld = {index + 1 for index, _flag in arg_credentials} | ({0} if command_credentials else set())

    # The program the launch really runs, with wrappers looked through ('env sh -c', 'timeout 30 bash -lc',
    # 'sudo sh -c', 'cmd /c sh -c'). A shell's '-c' program and an interpreter's inline program ('python -c',
    # 'node -e') are code, so their text is checked like a shell line.
    launched = unwrap_launch_command(_launch_argv(command, arg_list), expand_shell=False)
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
        metacharacters = (shell_text and _SHELL_METACHAR_RE.search(token)) or (
            not shell_text and (_SHELL_OPERATOR_ARG_RE.match(token.strip()) or _SUBSTITUTION_ARG_RE.search(token))
        )
        disables_tls = token in _INSECURE_TLS_FLAGS
        if not (metacharacters or disables_tls):
            continue
        shown = "<value withheld>" if index in withheld else _shown(token)
        if metacharacters:
            server.report(
                Severity.CRITICAL,
                "mcp_command_shell_metacharacters",
                f"command token contains shell metacharacters: {shown!r}",
                "Remove shell operators (; | & ` $() < >) from 'command' and from 'cmd /c' or PowerShell "
                "command lines; other arguments are passed argv-style, not through a shell.",
            )
        if disables_tls:
            server.report(
                Severity.CRITICAL,
                "mcp_command_disables_tls",
                f"command disables TLS/certificate verification: {shown!r}",
                "Remove insecure-TLS flags; do not disable certificate verification.",
            )
    for program in programs:
        # An inline program split out of one command-line string ('env -S "sh -c ..."') is not a token of its own.
        if program not in tokens and _SHELL_METACHAR_RE.search(program):
            server.report(
                Severity.CRITICAL,
                "mcp_command_shell_metacharacters",
                f"inline program contains shell metacharacters: {_shown(program)!r}",
                "Invoke the server binary directly instead of passing it an inline program.",
            )

    messages = [
        f"command line argument {flag!r} carries an inline credential; only ${{ENV}} references are allowed"
        if flag is not None
        else "command line contains an inline credential (value withheld); only ${ENV} references are allowed"
        for _index, flag in command_credentials
    ]
    messages += [
        f"command argument {flag!r} carries an inline credential; only ${{ENV}} references are allowed"
        if flag is not None
        else f"command argument args[{index}] contains an inline credential (value withheld); only ${{ENV}} "
        "references are allowed"
        for index, flag in arg_credentials
    ]
    for message in messages:
        server.report(
            Severity.CRITICAL,
            "mcp_command_inline_secret",
            message,
            'Pass the secret by reference (e.g. "${MY_TOKEN}"); never inline a raw credential in args.',
        )

    # Shell interpreter invoked with an inline program string ('sh -c "..."', 'bash -lc', 'env sh -c', 'sudo sh -c').
    if runs_shell_program:
        shell, wrapper = launched[0], _launch_argv(command, arg_list)[0]
        through = "" if wrapper == shell else f" through {_shown(wrapper)!r}"
        server.report(
            Severity.CRITICAL,
            "mcp_command_dangerous_form",
            f"command invokes a shell interpreter with '-c' ({_shown(shell)!r}{through}); this executes an "
            "arbitrary program string",
            "Invoke the server binary directly instead of wrapping it in a shell '-c' string.",
        )


# --------------------------------------------------------------------------- #
# URLs                                                                        #
# --------------------------------------------------------------------------- #
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


def _resolve_url_references(server: _ServerFindings, url: str, label: str, client: McpClient) -> str | None:
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
        server.report(
            Severity.HIGH,
            "mcp_url_env_not_expanded",
            f"{label} {shown!r} uses '${{...}}' environment expansion, which Codex does not apply to plugin "
            f"MCP URLs; Codex {effect}",
            "Write the full https:// endpoint in a Codex plugin; '${VAR:-default}' expansion is a Claude Code feature.",
        )
        _check_url_inline_secrets(server, url, label, text=expanded)
        return None
    if unresolved and _env_controls_endpoint(expanded):
        variables = ", ".join(dict.fromkeys(unresolved))
        server.report(
            Severity.LOW,
            "mcp_url_env_unchecked",
            f"{label} {shown!r} takes its scheme or host from environment variable(s) {variables} with no "
            "default, so the endpoint the client reaches cannot be checked statically",
            'Give the variable a safe default ("${VAR:-https://host/mcp}") or document the value users must set.',
        )
        _check_url_inline_secrets(server, url, label, text=expanded)
        return None
    return expanded


def _check_oauth_urls(
    server: _ServerFindings, config: dict[str, Any], allowed_private_hosts: HostAllowlist, *, client: McpClient
) -> None:
    """Claude Code fetches OAuth server metadata from ``oauth.authServerMetadataUrl``: the same URL policy applies."""
    oauth = config.get("oauth")
    if not isinstance(oauth, dict):
        return
    url = oauth.get("authServerMetadataUrl")
    if isinstance(url, str) and url.strip():
        _check_endpoint_url(server, url, "oauth.authServerMetadataUrl", allowed_private_hosts, client=client)


def _check_url(
    server: _ServerFindings, config: dict[str, Any], allowed_private_hosts: HostAllowlist, *, client: McpClient
) -> None:
    url = config.get("url")
    if not isinstance(url, str) or not url.strip():
        server.report(
            Severity.HIGH,
            "mcp_url_empty",
            "runnable MCP 'url' must be a non-empty string",
            "Set 'url' to the server endpoint using a secure https:// (or wss://) URL.",
        )
        return
    _check_endpoint_url(server, url, "url", allowed_private_hosts, client=client)


def _check_endpoint_url(
    server: _ServerFindings, url: str, label: str, allowed_private_hosts: HostAllowlist, *, client: McpClient
) -> None:
    """Scheme, host, credential, and endpoint checks for one URL field of an MCP server (``label`` names it)."""
    display = url  # messages show the declared text, redacted
    if _URL_EXPANSION_RE.search(url):
        expanded = _resolve_url_references(server, url, label, client)
        if expanded is None:
            return
        url = expanded
    shown = redacted_url(display)  # messages never echo userinfo or query credentials
    try:
        # ``raw`` is how urllib (and Python clients) read the text; ``parsed`` is how
        # WHATWG clients (Node and Rust MCP clients) read it. They differ only for an
        # ambiguous URL (see url_ambiguities).
        raw = urlparse(url.strip())
        parsed = urlparse(whatwg_url(url))
    except ValueError:  # e.g. an unbalanced '[' in the authority
        server.report(
            Severity.HIGH,
            "mcp_url_malformed_authority",
            f"{label} could not be parsed (malformed authority)",
            "Use a valid host[:port] authority, e.g. https://host:443/path.",
        )
        return
    problems = url_ambiguities(url)
    if problems:
        server.report(
            Severity.HIGH,
            "mcp_url_malformed_authority",
            f"{label} contains {', and '.join(problems)}, so MCP clients and URL parsers disagree on where it "
            f"points: {shown!r} (WHATWG clients {_client_reading(url)})",
            "Write the URL with '//' after the scheme and without backslashes, whitespace, or control "
            "characters, e.g. https://host/path.",
        )
    scheme = (parsed.scheme or "").lower()
    # Inline credentials in userinfo/query are persisted verbatim; check them
    # independent of the scheme (secure https URLs are the common case).
    userinfo = _check_url_inline_secrets(server, display, label, text=url, ambiguous=bool(problems))
    # Host findings name the URL the way the client that connects reads it: for
    # 'https://169.254.169.254\\@pub.example/' that is the metadata host, not pub.example.
    # Text that the URL holds as a credential is never shown, even when a client reads it as the host.
    client_shown = redacted_url(whatwg_url(url)) if problems and not userinfo else shown
    if problems:
        # A Python client may still connect where urllib reads the host: classify that one too.
        raw_host = _safe_hostname(raw)
        if raw_host and raw_host != _safe_hostname(parsed):
            _check_endpoint(
                server, raw_host, allowed_private_hosts, label=label, shown=f"{shown} (as Python clients read it)"
            )
    if scheme in ALLOWED_MCP_URL_SCHEMES:
        # A secure scheme alone is not a usable endpoint: require a host to connect
        # to, and reject a malformed authority/port. Otherwise a URL like "https://"
        # or "wss://:443/" would pass validation yet never reach a server.
        try:
            host = parsed.hostname
            _ = parsed.port  # property access raises ValueError on a malformed port
        except ValueError:
            server.report(
                Severity.HIGH,
                "mcp_url_malformed_authority",
                f"{label} has a malformed authority/port: {shown!r}",
                "Use a valid host[:port] authority, e.g. https://host:443/path.",
            )
            return
        if not host:
            server.report(
                Severity.HIGH,
                "mcp_url_no_host",
                f"{label} uses scheme {scheme!r} but has no host to connect to: {shown!r}",
                "Provide a full endpoint with a hostname, e.g. https://host[:port]/path.",
            )
        _check_endpoint(server, host, allowed_private_hosts, label=label, shown=client_shown)
        return
    if scheme in _INSECURE_URL_SCHEMES:
        # Plaintext endpoints are blocked below; still report where they point.
        _check_endpoint(server, _safe_hostname(parsed), allowed_private_hosts, label=label, shown=client_shown)
    if scheme == "":
        server.report(
            Severity.HIGH,
            "mcp_url_scheme_missing",
            f"{label} {shown!r} has no scheme, so MCP clients cannot connect to it",
            "Write the full endpoint with its scheme, e.g. https://host/mcp.",
        )
    elif scheme in _DANGEROUS_URL_SCHEMES:
        server.report(
            Severity.CRITICAL,
            "mcp_url_dangerous_scheme",
            f"{label} uses a dangerous scheme {scheme!r}: {shown!r}",
            "Use a secure https:// or wss:// endpoint; file/data/javascript/ftp schemes are not permitted.",
        )
    elif scheme in _INSECURE_URL_SCHEMES:
        # Same rule as HTTP hooks: plaintext to this machine's loopback interface does not cross a network
        # (a local desktop app's MCP server); the MEDIUM loopback finding above still reports it.
        endpoint = classify_endpoint_host(_safe_hostname(parsed) or "")
        if endpoint is None or not endpoint.is_loopback:
            server.report(
                Severity.HIGH,
                "mcp_url_insecure_scheme",
                f"{label} uses an insecure plaintext scheme {scheme!r}: {shown!r}",
                "Use https:// (or wss://) so the MCP transport is encrypted.",
            )
    else:
        server.report(
            Severity.HIGH,
            "mcp_url_scheme_not_allowed",
            f"{label} scheme {scheme!r} is not an allowed MCP scheme: {shown!r}",
            f"Use one of the allowed secure schemes: {', '.join(sorted(ALLOWED_MCP_URL_SCHEMES))}.",
        )


# --------------------------------------------------------------------------- #
# env, headers, transport, and TLS settings                                   #
# --------------------------------------------------------------------------- #
def _check_env_and_headers(server: _ServerFindings, config: dict[str, Any], *, client: McpClient) -> None:
    for section in ("env", "headers"):
        block = config.get(section)
        if block is None:
            continue
        if not isinstance(block, dict):
            server.report(
                Severity.HIGH,
                "mcp_env_not_object",
                f"'{section}' must be an object mapping names to reference values",
                f"Express '{section}' as a JSON object of key -> value.",
            )
            continue
        # NON-BLOCKING advisory: the wrapper runtime applies command+args (stdio) and
        # url (http/sse) only -- Harbor's per-MCP-server config has no env/headers field.
        # Only a native Claude Code arm (the plugin's own .mcp.json) applies them. The
        # inline-secret / insecure-TLS checks below still run, so a raw credential
        # declared here is still caught and blocks.
        server.report(
            Severity.LOW,
            "mcp_field_ignored",
            f"'{section}' is applied only when Tier 3 loads the plugin natively in Claude Code; the wrapper "
            "runtime and the other harnesses ignore it, and such a Tier 3 run of this server is reported "
            "INCOMPLETE",
            f"Use --plugin-load native with Claude Code, or remove '{section}' and rely on task-level "
            "environment / CI credential injection.",
        )
        _check_secret_values(server, section, block, tls_env=True)
    oauth = config.get("oauth")
    if isinstance(oauth, dict):
        # A literal client secret in the OAuth block ships with the plugin like any other credential.
        _check_secret_values(server, "oauth", oauth, tls_env=False)
    if client == "codex":
        for field_name in CODEX_UNAPPLIED_MCP_FIELDS:
            if config.get(field_name):
                server.report(
                    Severity.LOW,
                    "mcp_field_ignored",
                    f"Codex '{field_name}' is not applied by the evaluation runtime; a Tier 3 run of this server "
                    "is reported INCOMPLETE",
                    f"Expect an INCOMPLETE Tier 3 run, or remove '{field_name}' if the server works without it.",
                )


def _check_secret_values(server: _ServerFindings, section: str, block: dict[Any, Any], *, tls_env: bool) -> None:
    """Inline credentials (and, for env and headers, TLS-off switches) among one block's string values."""
    for key, value in block.items():
        if not isinstance(value, str):
            continue
        if tls_env and _is_insecure_tls_env(key, value):
            server.report(
                Severity.CRITICAL,
                "mcp_insecure_tls_env",
                f"'{section}.{key}' disables TLS/certificate verification",
                "Do not disable TLS verification via environment variables.",
            )
        # A header name such as 'Key' names a credential the way a query key does; an env name 'KEY' needs a
        # value shaped like key material.
        if looks_like_inline_secret(key, value, query=section == "headers"):
            server.report(
                Severity.CRITICAL,
                "mcp_inline_secret",
                f"'{section}.{key}' contains an inline credential; only ${{ENV}} references are allowed",
                'Reference a secret handle/env var (e.g. "${MY_TOKEN}"); never inline a raw secret.',
            )
        elif looks_like_random_secret(key, env_defaults_text(value)):
            server.report(
                Severity.MEDIUM,
                "mcp_possible_inline_secret",
                f"'{section}.{key}' holds a long random-looking value that may be a credential (value withheld); "
                "the key name does not say what it is",
                'If it is a secret, reference it (e.g. "${MY_SECRET}"); if not, rename the key to say what it holds.',
            )


def _check_transport(server: _ServerFindings, config: dict[str, Any]) -> None:
    raw = config.get("transport", config.get("type"))
    if raw is None:
        return
    if not isinstance(raw, str) or raw.strip().lower() not in ALLOWED_MCP_TRANSPORTS:
        server.report(
            Severity.HIGH,
            "mcp_transport_invalid",
            f"transport {raw!r} is not one of {sorted(ALLOWED_MCP_TRANSPORTS)}",
            f"Set transport to one of: {', '.join(sorted(ALLOWED_MCP_TRANSPORTS))}.",
        )
        return

    literal = raw.strip()
    canonical = literal.lower()
    # Harbor's transport literal is case-sensitive: the agent adapter compares it
    # against the exact lowercase "stdio"/"http"/"sse" and the persist path writes
    # it verbatim, so a value Tier 1 accepts must be the exact form Harbor accepts.
    if literal != canonical:
        server.report(
            Severity.HIGH,
            "mcp_transport_bad_casing",
            f"transport {raw!r} must be lowercase {canonical!r}; Harbor's transport literal is case-sensitive",
            f"Use the exact lowercase transport literal {canonical!r}.",
        )

    # Kind <-> transport consistency: a stdio server is launched from a 'command';
    # an http/sse server is reached over a 'url'. Harbor rejects a transport that
    # contradicts the declared kind (http/sse need a url; stdio needs a command).
    has_command = "command" in config
    has_url = "url" in config
    if has_command and not has_url and canonical != "stdio":
        server.report(
            Severity.HIGH,
            "mcp_transport_kind_mismatch",
            f"command (stdio) server declares transport {raw!r}; a command server must use transport 'stdio'",
            "Set transport to 'stdio' (or omit it) for command-based MCP servers.",
        )
    elif has_url and not has_command and canonical not in {"http", "sse"}:
        server.report(
            Severity.HIGH,
            "mcp_transport_kind_mismatch",
            f"url server declares transport {raw!r}; a url server must use transport 'http' or 'sse'",
            "Set transport to 'http' or 'sse' for url-based MCP servers.",
        )


def _check_insecure_tls_config(server: _ServerFindings, config: dict[str, Any]) -> None:
    """Reject config keys that turn off TLS/certificate verification."""
    if config.get("insecure") is True:
        server.report(
            Severity.CRITICAL,
            "mcp_insecure_flag",
            "'insecure: true' disables endpoint security",
            "Remove 'insecure'; connect over a verified TLS endpoint.",
        )
    for section in ("tls", "ssl"):
        block = config.get(section)
        if not isinstance(block, dict):
            continue
        if block.get("rejectUnauthorized") is False or block.get("verify") is False:
            server.report(
                Severity.CRITICAL,
                "mcp_insecure_tls_config",
                f"'{section}' disables certificate verification (rejectUnauthorized/verify = false)",
                "Do not disable certificate verification; use a valid certificate chain.",
            )


# --------------------------------------------------------------------------- #
# Supply-chain pinning                                                        #
# --------------------------------------------------------------------------- #
PinStatus = Literal["pinned", "unpinned", "not_applicable"]
RunnerEcosystem = Literal["npm", "pypi", "deno", "go", "nuget", "container"]

# An exact semantic version, with optional prerelease and build metadata.
_SEMVER = r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?"
# The exact-version matchers are shared with the dependency audit, so a runner
# spec counts as pinned exactly when the audit can match it to one release.
# An exact npm version: "1.2.3", "=1.2.3", or "v1.2.3", with optional
# prerelease and build metadata.
_NPM_EXACT_RE = re.compile(rf"^=?v?({_SEMVER})$")
# An exact Go module version ('go run mod@v1.2.3') and an exact .NET tool version ('dnx Pkg@1.2.3').
_GO_EXACT_RE = re.compile(rf"^v{_SEMVER}$")
_NUGET_EXACT_RE = re.compile(rf"^v?{_SEMVER}$")
# The canonical PEP 440 version pattern, verbatim from PEP 440 Appendix B. It
# matches every spelling PEP 440 accepts for one version, such as "1.2.3",
# "v1.2.3", "1.0rc", "1.0-1", or "1.2.3-beta.1"; a wildcard such as "1.0.*" is
# not a version.
_PEP440_VERSION_PATTERN = r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""
PEP440_VERSION_RE = re.compile(r"^\s*" + _PEP440_VERSION_PATTERN + r"\s*$", re.VERBOSE | re.IGNORECASE)
# A PyPI requirement: a distribution name, optional extras, then its version specifier.
_PYPI_REQUIREMENT_RE = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?\s*(?:\[[^\]]*\])?\s*(?P<specifier>.*)", re.DOTALL
)
# A version specifier that can pin one version: "==V", "===V", or uv's "@V".
_PIN_SPECIFIER_RE = re.compile(r"(?P<operator>===|==|@)\s*(?P<version>\S+)")
_GIT_SHA_RE = re.compile(r"(?:#|@)[0-9a-fA-F]{40}(?:$|[&#])")
_DOCKER_DIGEST_RE = re.compile(r"@sha256:[0-9a-fA-F]{64}$")
# A dot belongs to the segment body, so only "-", "+" or "_" can start a later
# segment. The old "(?:[-+._][0-9A-Za-z.]+)*" accepted the same tags but could
# split a run of ".x" in exponentially many ways (ReDoS on a long bad tag).
_VERSION_TAG_RE = re.compile(r"^v?\d+(?:\.\d+)*(?:[-+._][0-9A-Za-z.]+(?:[-+_][0-9A-Za-z.]+)*)?$")
_DOTTED_VERSION_PREFIX_RE = re.compile(r"^v?\d+\.\d+(?![0-9A-Za-z])")
# A remote module path that names an exact version (``/x/mod@v1.2.3/mod.ts``).
_DENO_EXACT_MODULE_PATH_RE = re.compile(r"@v?\d+\.\d+\.\d+(?:/|$)")
_IMAGE_ENV_REF_RE = re.compile(r"\$\{[^}]*\}|\$[A-Za-z_][A-Za-z0-9_]*")
_PLUGIN_PATH_REFS: tuple[str, ...] = ("${CLAUDE_PLUGIN_ROOT}", "${CLAUDE_PLUGIN_DATA}", "${CLAUDE_PROJECT_DIR}")
_LOCAL_SPEC_PREFIXES: tuple[str, ...] = (".", "/", "~", "file:", *_PLUGIN_PATH_REFS)
_REMOTE_SPEC_PREFIXES: tuple[str, ...] = ("git+", "git:", "github:", "gitlab:", "bitbucket:", "http://", "https://")

# Value-taking flags per package runner, so a flag's value is never mistaken for
# the package spec. The npm runners read an option missing from both their value
# flags and _NPM_SWITCHES fail-closed (see _npm_invocation); the other runners
# read it as a switch.
_NPX_VALUE_FLAGS = frozenset(
    {
        *("-p", "--package", "-c", "--call", "--registry", "--cache", "--userconfig", "--globalconfig"),
        *("--prefix", "-C", "-w", "--workspace", "--loglevel", "--node-options", "--script-shell", "--shell"),
        *("--location", "-L", "--before", "--tag", "--omit", "--include", "-n", "--node-arg", "--npm"),
    }
)
_DLX_VALUE_FLAGS = frozenset(
    {"-p", "--package", "--registry", "--allow-build", "-C", "--dir", "--reporter", "--loglevel", "--filter"}
)
# Options that take a value before the 'exec' subcommand of npm.
_NPM_GLOBAL_VALUE_FLAGS = frozenset({*_NPX_VALUE_FLAGS, "--dir"})
# Options of the npm runners that take no value: npx's own switches, npm's Boolean
# options and the shorthands that expand to one, and the bunx and pnpm dlx switches.
# A runner's value flags win, so npx's '-c' (--call) still takes a value.
_NPM_SWITCHES = frozenset(
    {
        *("-y", "--yes", "--no-install", "--ignore-existing", "--always-spawn", "--shell-auto-fallback"),
        *("-q", "--quiet", "-s", "--silent", "-d", "-dd", "-ddd", "--verbose", "-g", "--global", "-f", "--force"),
        *("--offline", "--prefer-offline", "--prefer-online", "--ignore-scripts", "--foreground-scripts"),
        *("--legacy-peer-deps", "--strict-peer-deps", "--install-links", "--package-lock", "--strict-ssl"),
        *("--dry-run", "--json", "--parseable", "-l", "--long", "-ws", "--workspaces", "-iwr"),
        *("--include-workspace-root", "--audit", "--fund", "--progress", "--color", "--unicode", "--timing"),
        *("--update-notifier", "-v", "--version", "-h", "--help", "--usage", "--bun", "-c", "--shell-mode"),
    }
)
_PACKAGE_FLAGS = frozenset({"-p", "--package"})
_WITH_FLAGS = ("--with", "-w")
_UVX_VALUE_FLAGS = frozenset(
    {
        "--from",
        "--with",
        "-w",
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
        "-b",
        "--config-setting",
        "-C",
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
# Options of 'uv' and 'uv run' that take a value, before 'run' and before the script or command it runs.
_UV_RUN_VALUE_FLAGS = frozenset(
    {
        "--with",
        "-w",
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
_PIPX_VALUE_FLAGS = frozenset({"--spec", "--python", "--pip-args", "--index-url", "--fetch-python"})
_DENO_VALUE_FLAGS = frozenset({"-c", "--config", "--import-map", "--lock", "--cert", "--location", "--seed"})
_GO_RUN_VALUE_FLAGS = frozenset({"-C", "-exec", "-o", "-p", "-tags", "-ldflags", "-gcflags", "-mod", "-modfile"})
_DNX_VALUE_FLAGS = frozenset({"--version", "--source", "--add-source", "--configfile", "--verbosity", "-v"})
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
# Options of a container runtime that take a value before its 'run' subcommand.
_CONTAINER_GLOBAL_VALUE_FLAGS = frozenset({"-c", "--context", "-H", "--host", "--config", "-l", "--log-level"})


# --------------------------------------------------------------------------- #
# Launch commands                                                             #
# --------------------------------------------------------------------------- #
# env options (GNU and BSD) whose value is the rest of the word or the next argument; '-S' /
# '--split-string' also splits its value into the arguments env reads next.
_ENV_VALUE_LETTERS = frozenset("uCPaLU")
_ENV_VALUE_LONG_OPTIONS: tuple[str, ...] = ("unset", "chdir", "argv0")
_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


@dataclass(frozen=True)
class _Wrapper:
    """A program that runs the command after its own options: ``nohup cmd``, ``timeout 60 cmd``."""

    # Options that take the next word as their value.
    value_options: frozenset[str] = frozenset()
    # Words between the options and the command: timeout's duration.
    operands: int = 0
    # NAME=value words before the command set its environment (sudo).
    assignments: bool = False

    def command(self, words: list[str]) -> list[str]:
        """The command line this wrapper runs, given the wrapper's own arguments."""
        index = 0
        while index < len(words) and words[index].startswith("-") and words[index] != "-":
            if words[index] == "--":
                index += 1
                break
            index += 2 if words[index] in self.value_options else 1
        while self.assignments and index < len(words) and _ENV_ASSIGNMENT_RE.match(words[index]):
            index += 1
        return words[index + self.operands :]


# Programs that run the command after their options; env, which also reads NAME=value
# assignments and a '-S' string, is read by _env_command.
_WRAPPERS: dict[str, _Wrapper] = {
    "nohup": _Wrapper(),
    "setsid": _Wrapper(),
    "nice": _Wrapper(frozenset({"-n", "--adjustment"})),
    "stdbuf": _Wrapper(frozenset({"-i", "--input", "-o", "--output", "-e", "--error"})),
    "time": _Wrapper(frozenset({"-f", "--format", "-o", "--output"})),
    "timeout": _Wrapper(frozenset({"-k", "--kill-after", "-s", "--signal"}), operands=1),
    "sudo": _Wrapper(
        frozenset(
            {
                *("-u", "--user", "-g", "--group", "-U", "--other-user", "-C", "--close-from", "-D", "--chdir"),
                *("-h", "--host", "-p", "--prompt", "-r", "--role", "-t", "--type", "-T", "--command-timeout"),
            }
        ),
        assignments=True,
    ),
    "doas": _Wrapper(frozenset({"-u", "-C", "-a"})),
    "exec": _Wrapper(frozenset({"-a"})),
    "command": _Wrapper(),
    "busybox": _Wrapper(),
}
# Shell options whose value is the program, and long options that take a value.
_SHELL_COMMAND_OPTIONS = frozenset({"--command"})
_SHELL_LONG_VALUE_OPTIONS = frozenset({"--rcfile", "--init-file", "--init-command"})
# fish runs a program from '-c' / '--command' and from '-C' / '--init-command' (before its
# script), and its '-d', '-o', '-f', '-p' (and their long forms) take a value.
_FISH_INLINE_PROGRAM_LONG_OPTIONS: tuple[str, ...] = ("command", "init-command")
_FISH_VALUE_LETTERS = frozenset("dofp")
_FISH_VALUE_LONG_OPTIONS: tuple[str, ...] = (
    "debug",
    "debug-output",
    "debug-stack-frames",
    "features",
    "profile",
    "profile-startup",
)
_MAX_UNWRAP_DEPTH = 8
# Every program the launch reader knows by name, so a 'command' with spaces that names one in its last path
# segment can be a path to it ('C:\\Program Files\\nodejs\\npx.cmd'); see _launch_argv.
_KNOWN_LAUNCHERS = frozenset(
    {
        *("npx", "bunx", "pnpx", "bun", "pnpm", "yarn", "npm", "uvx", "uv", "pipx", "deno", "go", "dnx"),
        *("cmd", "powershell", "pwsh", "env"),
        *_CONTAINER_RUNTIMES,
        *_SHELL_INTERPRETERS,
        *_WRAPPERS,
    }
)


def _command_basename(command: str) -> str:
    base = command.strip().strip("\"'").replace("\\", "/").rsplit("/", 1)[-1].lower()
    for suffix in (".exe", ".cmd", ".bat", ".ps1"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _split_command_line(tokens: list[str]) -> list[str]:
    """A command line given as one string (``cmd /c "npx -y pkg"``) as words; other argv as is."""
    if len(tokens) == 1 and any(char.isspace() for char in tokens[0]):
        return [word.strip("\"'") for word in tokens[0].split()]
    return list(tokens)


def _launch_argv(command: str, args: list[str]) -> list[str]:
    """The argv a launch config runs: ``command`` plus ``args``.

    ``command`` may name only the program, possibly as a path with spaces such as
    ``C:\\Program Files\\nodejs\\npx.cmd``, so every option is in ``args``; or it
    may hold a whole command line (``npx -y pkg``, ``bash -c node``), which is
    split into words. Its first word decides. The string is one program only when
    it reads as a path with spaces (its first word has a ``/`` or ``\\``, and no
    later word is an option) whose first word names no known launcher, while the
    whole string does. So ``npx -y pkg /srv/docker`` and ``bash -c x
    /usr/bin/env`` are command lines, whatever their last path segment names.
    """
    stripped = command.strip()
    words = _split_command_line([stripped])
    path_with_spaces = (
        ("/" in words[0] or "\\" in words[0])
        and not any(word.startswith("-") for word in words[1:])
        and _command_basename(words[0]) not in _KNOWN_LAUNCHERS
        and _command_basename(stripped) in _KNOWN_LAUNCHERS
    )
    if len(words) > 1 and not path_with_spaces:
        return [*words, *args]
    return [stripped, *args]


def unwrap_launch_command(tokens: list[str], *, expand_shell: bool = True, stop_at: Iterable[str] = ()) -> list[str]:
    """Strip launch wrappers so the runner they start is classified.

    ``cmd /c <command line>``, ``powershell -Command <line>``, ``env`` (with its
    options, ``NAME=value`` assignments, and ``-S`` string; see
    :func:`_env_command`), and the programs in ``_WRAPPERS`` (``timeout
    <duration>``, ``nice``, ``nohup``, ``setsid``, ``sudo``, ``doas``, ``exec``,
    ``command``, ``time``, ``stdbuf``, ``busybox``) are removed, along with their
    own options. With ``expand_shell``, ``sh -c '<program>'`` is replaced by the
    program's words (its first command only; hook analysis splits shell programs
    into commands itself and passes ``False``). Unwrapping stops at a launcher
    named in ``stop_at``. The result is empty when a wrapper names no command.
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
        if base in _COMMAND_LINE_PARSERS:
            switches = _COMMAND_LINE_PARSERS[base]
            switch = next((index for index, token in enumerate(rest) if token.lower() in switches), None)
            if switch is None:
                return current
            line = rest[switch + 1 :]
            if base == "cmd":
                current = _split_command_line(line)
            else:
                current = _split_command_line([" ".join(line)]) if line else []
        elif base in _SHELL_INTERPRETERS and expand_shell:
            program = _shell_program(current)
            if program is None:
                return current
            current = _split_command_line([program]) if program.strip() else []
        elif base == "env":
            current = _env_command(rest)
        elif base in _WRAPPERS:
            current = _WRAPPERS[base].command(rest)
        else:
            return current
    return current


def _env_command(words: list[str]) -> list[str]:
    """The command line ``env`` runs, given env's arguments: ``env -u X A=1 sh -c y`` runs ``sh -c y``."""
    index = 0
    while index < len(words):
        word = words[index]
        index += 1
        if word == "--":
            break
        if word == "-" or _ENV_ASSIGNMENT_RE.match(word):  # '-' is the old spelling of '-i'
            continue
        if not word.startswith("-"):
            return words[index - 1 :]
        takes_value, splits, value = _env_option(word)
        if takes_value and value is None and index < len(words):
            value = words[index]
            index += 1
        if splits and value:
            # -S splits its value into arguments that env reads in its place, options and all.
            words, index = [*_split_words(value), *words[index:]], 0
    return words[index:]


def _env_option(word: str) -> tuple[bool, bool, str | None]:
    """``(takes a value, splits the value into arguments, the value attached to the word)`` for an env option."""
    if word.startswith("--"):
        # GNU env accepts any unambiguous prefix of a long option ('--split' for '--split-string').
        name, equals, attached = word[2:].partition("=")
        splits = bool(name) and "split-string".startswith(name)
        takes_value = splits or (bool(name) and any(option.startswith(name) for option in _ENV_VALUE_LONG_OPTIONS))
        return takes_value, splits, (attached if equals else None)
    for position, letter in enumerate(word[1:], start=1):
        if letter == "S" or letter in _ENV_VALUE_LETTERS:
            return True, letter == "S", word[position + 1 :] or None
    return False, False, None


def _split_words(text: str) -> list[str]:
    """Shell words of ``text`` (quotes removed); whitespace split when the quoting is unbalanced."""
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def _shell_inline(tokens: list[str]) -> tuple[bool, str | None]:
    """``(runs a -c program, the program)`` for a shell argv (``tokens[0]`` is the shell).

    Reads ``sh -c '<program>'``, combined flags (``bash -lc``, ``sh -ec``), the
    ``+c`` form bash, sh, and zsh also accept, options before or after ``-c``
    (``bash -o pipefail -c``, ``bash -oe pipefail -c``, ``bash -c -e``), ``--``
    before the program, and ``--command``. fish also runs a program from ``-C`` /
    ``--init-command`` (before its script) and accepts any unambiguous prefix of a
    long option; its arguments are read both ways, so the fish reading can only
    flag more.
    """
    if not tokens:
        return False, None
    found = _read_inline_program(tokens[1:], _shell_option)
    if not found[0] and _command_basename(tokens[0]) == "fish":
        found = _read_inline_program(tokens[1:], _fish_option)
    return found


def _read_inline_program(args: list[str], read_option: Callable[[str], tuple[bool, bool]]) -> tuple[bool, str | None]:
    """``(runs an inline program, the program)`` for a shell's arguments, its options read with ``read_option``.

    A shell reads options only until its first operand (the script it runs) or
    an end-of-options marker (``--`` or ``-``), so a script's own ``-config`` is
    not the shell's ``-c``. The program is the value of ``--command=<program>``,
    else the first word after the options; it is ``None`` when none follows.
    """
    inline = False
    index = 0
    while index < len(args):
        arg = args[index].strip()
        if arg == "--":
            index += 1
            break
        if arg == "-" or not arg.startswith(("-", "+")):
            break
        runs, takes_value = read_option(arg)
        if runs and arg.startswith("--") and "=" in arg:
            return True, arg.partition("=")[2]
        inline = inline or runs
        index += 2 if takes_value else 1
    if not inline:
        return False, None
    return True, (args[index] if index < len(args) else None)


def _shell_option(arg: str) -> tuple[bool, bool]:
    """``(runs an inline program, takes the next argument as its value)`` for one sh/bash/zsh/dash/ksh option.

    ``-c`` alone or inside a short-option cluster (``-lc``, ``-ec``), its ``+c``
    form, and ``--command`` run one. ``-o name`` / ``-O name`` (also ``+o``,
    ``+O``, and inside a cluster such as ``-eo pipefail``), ``--rcfile``,
    ``--init-file``, and ``--init-command`` take a value.
    """
    if arg.startswith("--"):
        name, equals, _value = arg.partition("=")
        return name in _SHELL_COMMAND_OPTIONS, not equals and name in _SHELL_LONG_VALUE_OPTIONS
    letters = arg[1:]
    return "c" in letters, "o" in letters or "O" in letters


def _fish_option(arg: str) -> tuple[bool, bool]:
    """``(runs an inline program, takes the next argument as its value)`` for one fish option."""
    if arg.startswith("--"):
        # fish accepts any unambiguous prefix of a long option; an ambiguous one is an error, flagged anyway.
        name, equals, _value = arg[2:].partition("=")
        if name and any(option.startswith(name) for option in _FISH_INLINE_PROGRAM_LONG_OPTIONS):
            return True, False
        return False, not equals and bool(name) and any(option.startswith(name) for option in _FISH_VALUE_LONG_OPTIONS)
    letters = arg[1:]
    if "c" in letters or "C" in letters:
        return True, False
    # In a cluster, a value option takes the rest of the word, or the next argument when it ends the word.
    return False, letters[-1:] in _FISH_VALUE_LETTERS


def _shell_program(tokens: list[str]) -> str | None:
    """The program string of ``sh -c '<program>'`` (also ``bash -lc``); ``None`` otherwise."""
    return _shell_inline(tokens)[1]


def shell_program(tokens: list[str]) -> str | None:
    """The program string when ``tokens`` run a shell interpreter with ``-c`` (``bash -c '<program>'``)."""
    if not tokens or _command_basename(tokens[0]) not in _SHELL_INTERPRETERS:
        return None
    return _shell_program([str(token) for token in tokens])


# --------------------------------------------------------------------------- #
# Package runners                                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class McpPinning:
    """Static supply-chain pinning classification of one MCP declaration.

    ``floating`` marks an ``unpinned`` launch whose version or tag names a
    moving release channel (``@latest``, ``:main``, ``:1-nightly``); it is the
    blocking case. Other ``unpinned`` launches (no version, a range) warn.
    ``remote`` is set when the package comes from a git or URL spec, or a
    remote module, rather than from a registry.
    """

    status: PinStatus
    detail: str
    floating: bool = False
    remote: bool = False

    @property
    def pinned(self) -> bool | None:
        """``True``/``False`` for package runners; ``None`` when not applicable."""
        if self.status == "not_applicable":
            return None
        return self.status == "pinned"


@dataclass(frozen=True)
class RunnerInvocation:
    """A package runner that an MCP server command runs, and the specs it fetches.

    ``runner`` names the invocation as findings show it (``npx``, ``pnpm dlx``,
    ``uv tool run``, ``docker run``). The command is read through launch
    wrappers (:func:`unwrap_launch_command`), and ``specs`` are read from the
    argv the way the runner reads it:

    * ``npm`` (``npx``, ``bunx``, ``bun x``, ``pnpx``, ``pnpm dlx``, ``yarn
      dlx``, ``npm exec``): every ``-p``/``--package`` value (also one given
      before ``dlx``, and each item of ``a,b``), else the first positional
      argument;
    * ``pypi`` (``uvx``, ``uv tool run``, ``pipx run``, ``uv run``): the
      ``--from`` or ``--spec`` requirement, else the first positional argument,
      then every ``--with`` (``-w``) requirement, which ``with_specs`` also
      lists (``uv run`` installs only those, next to the project environment);
      the files of ``--with-requirements`` are ``requirement_files``, whose
      packages are not read;
    * ``deno`` (``deno run``): the module it runs, an ``npm:`` or ``jsr:``
      spec, a URL, or a local script;
    * ``go`` (``go run``): the package or module it runs, with its ``@version``;
    * ``nuget`` (``dnx``): the .NET tool package, as ``Package@version`` when
      ``--version`` names the version;
    * ``container`` (``docker``, ``podman``, or ``nerdctl run``): the image.

    A runner's options end at the package, command, module, or image it runs
    (``npm exec`` reads them up to ``--``); later arguments belong to the
    server. ``specs`` is empty when the runner names no package or image.
    """

    ecosystem: RunnerEcosystem
    runner: str
    specs: tuple[str, ...]
    requirement_files: tuple[str, ...] = ()
    with_specs: tuple[str, ...] = ()

    @property
    def npm_specs(self) -> tuple[str, ...]:
        """The npm registry specs the runner installs, including the package of ``deno run npm:pkg``."""
        if self.ecosystem == "npm":
            return self.specs
        if self.ecosystem == "deno":
            return tuple(spec.removeprefix("npm:") for spec in self.specs if spec.startswith("npm:"))
        return ()


def _argv(config: Any) -> list[str] | None:
    """The argv a runnable MCP declaration runs, through its launch wrappers, or ``None`` when it has no command.

    A wrapper that names no command (``env`` alone) is the program itself.
    """
    if not isinstance(config, dict):
        return None
    command = config.get("command")
    if not isinstance(command, str) or not command.strip():
        return None
    raw_args = config.get("args")
    args = [str(arg) for arg in raw_args] if isinstance(raw_args, list) else []
    argv = _launch_argv(command, args)
    return unwrap_launch_command(argv) or argv


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


def _subcommand(tokens: list[str], value_flags: frozenset[str]) -> tuple[str | None, list[str]]:
    """``(subcommand, tokens after it)`` of a runner front end, skipping its global options."""
    index = _skip_options(tokens, value_flags)
    if index >= len(tokens):
        return None, []
    return tokens[index], tokens[index + 1 :]


def _runner_options(args: list[str], value_flags: frozenset[str]) -> tuple[list[str], str | None]:
    """A runner's own options, and its first positional argument (the package or command it runs).

    The runner reads options only up to that argument (or ``--``): every later
    argument belongs to the server, so a server's ``-p 3000`` or ``--with x``
    is never read as the runner's.
    """
    first = next(_positionals(args, value_flags), None)
    if first is None:
        return args, None
    index, positional = first
    return args[:index], positional


def _package_argument(options: list[str], positional: str | None, flag: str) -> str | None:
    """The value of a runner's package option (``--from``, ``--spec``), else its first positional argument."""
    values = _flag_values(options, (flag,))
    return values[0] if values else positional


def _with_requirements(options: list[str]) -> tuple[str, ...]:
    """Every ``--with`` (``-w``) requirement in ``options``; one value may list several, comma-separated."""
    items = (item.strip() for value in _flag_values(options, _WITH_FLAGS) for item in value.split(","))
    return tuple(item for item in items if item)


def parse_mcp_runner(config: Any) -> RunnerInvocation | None:
    """The package runner an MCP declaration runs, or ``None`` when it runs none.

    This is the one argv reader for package runners: the pinning check
    (:func:`classify_mcp_pinning`), the container-image lookup
    (:func:`mcp_container_image`), and the dependency audit all read runner
    invocations through it, so they agree on what a server installs. Launch
    wrappers (``cmd /c``, ``env X=1``, ``timeout``, ``sudo``, ``sh -c``) are
    looked through. Local interpreters, scripts, and binaries, container
    subcommands other than ``run``, ``uv run`` without ``--with``, URL servers,
    and provider-only entries give ``None``.
    """
    argv = _argv(config)
    return None if argv is None else _runner_invocation(argv)


def _runner_invocation(argv: list[str]) -> RunnerInvocation | None:
    base, args = _command_basename(argv[0]), argv[1:]
    if base in {"npx", "bunx", "pnpx"}:
        return _npm_invocation(base, args, _NPX_VALUE_FLAGS if base == "npx" else _DLX_VALUE_FLAGS)
    if base == "bun":
        sub, rest = _subcommand(args, _DLX_VALUE_FLAGS)
        return _npm_invocation("bun x", rest, _DLX_VALUE_FLAGS) if sub == "x" else None
    if base in {"pnpm", "yarn"}:
        sub, rest = _subcommand(args, _DLX_VALUE_FLAGS)
        if sub != "dlx":
            return None
        # '--package' may come before 'dlx' ('pnpm --package=@scope/pkg dlx bin').
        before = args[: len(args) - len(rest) - 1]
        packages = [f"--package={spec}" for spec in _flag_values(before, _PACKAGE_FLAGS)]
        return _npm_invocation(f"{base} dlx", [*packages, *rest], _DLX_VALUE_FLAGS)
    if base == "npm":
        sub, rest = _subcommand(args, _NPM_GLOBAL_VALUE_FLAGS)
        if sub not in {"exec", "x"}:
            return None
        # Like every npm command, npm exec reads its options anywhere before "--".
        return _npm_invocation("npm exec", rest, _NPX_VALUE_FLAGS, options_until_separator=True)
    if base == "uvx":
        return _uv_invocation("uvx", args)
    if base == "uv":
        sub, rest = _subcommand(args, _UV_RUN_VALUE_FLAGS)
        if sub == "tool":
            tool_sub, tool_rest = _subcommand(rest, _UVX_VALUE_FLAGS)
            return _uv_invocation("uv tool run", tool_rest) if tool_sub in {"run", "x"} else None
        if sub == "run":
            # 'uv run' runs the project environment; '--with' adds packages resolved on every run.
            options = [*args[: len(args) - len(rest) - 1], *rest[: _skip_options(rest, _UV_RUN_VALUE_FLAGS)]]
            extras = _with_requirements(options)
            files = tuple(_flag_values(options, ("--with-requirements",)))
            if not extras and not files:
                return None
            return RunnerInvocation("pypi", "uv run", extras, requirement_files=files, with_specs=extras)
        return None
    if base == "pipx" and args[:1] == ["run"]:
        options, app = _runner_options(args[1:], _PIPX_VALUE_FLAGS)
        spec = _package_argument(options, app, "--spec")
        return RunnerInvocation("pypi", "pipx run", () if spec is None else (spec,))
    if base == "deno" and args[:1] == ["run"]:
        module = _first_positional(args[1:], _DENO_VALUE_FLAGS)
        return RunnerInvocation("deno", "deno run", () if module is None else (module,))
    if base == "go" and args[:1] == ["run"]:
        package = _first_positional(args[1:], _GO_RUN_VALUE_FLAGS)
        return RunnerInvocation("go", "go run", () if package is None else (package,))
    if base == "dnx":
        package = _first_positional(args, _DNX_VALUE_FLAGS)
        versions = _flag_values(args, ("--version",))
        if package is not None and versions:
            package = f"{package}@{versions[0]}"
        return RunnerInvocation("nuget", "dnx", () if package is None else (package,))
    if base in _CONTAINER_RUNTIMES:
        sub, rest = _subcommand(args, _CONTAINER_GLOBAL_VALUE_FLAGS)
        if sub == "container" and rest[:1] == ["run"]:
            rest = rest[1:]
        elif sub != "run":
            return None
        image = _first_positional(rest, _DOCKER_VALUE_FLAGS)
        return RunnerInvocation("container", f"{base} run", () if image is None else (image,))
    return None


def _npm_invocation(
    runner: str, args: list[str], value_flags: frozenset[str], *, options_until_separator: bool = False
) -> RunnerInvocation:
    """An npm package runner: every ``-p``/``--package`` value, else the first positional argument.

    The runner's options end at the package it runs, or, with
    *options_until_separator*, at ``--``. A value flag takes the next word and
    a switch (``_NPM_SWITCHES``) takes none. Any other option takes the next
    word unless it starts with ``-``, as npx reads it; but npm reads an option
    it does not know as a switch, so that word may be the package too and is
    kept as one more spec. The specs can only gain a package this way, never
    lose the one the runner installs. ``-p a@latest,b`` is read as two
    packages, so a floating one cannot hide behind a comma.
    """
    packages: list[str] = []
    maybe_packages: list[str] = []
    first: str | None = None
    index = 0
    while index < len(args):
        word = args[index]
        index += 1
        if word == "--":
            if first is None and index < len(args):
                first = args[index]
            break
        if word == "-" or not word.startswith("-"):
            if first is None:
                first = word
            if options_until_separator:
                continue
            break
        name, equals, value = word.partition("=")
        if equals:
            if name in _PACKAGE_FLAGS:
                packages.append(value)
        elif name in value_flags:
            if index < len(args) and name in _PACKAGE_FLAGS:
                packages.append(args[index])
            index += 1
        elif name not in _NPM_SWITCHES and index < len(args) and not args[index].startswith("-"):
            maybe_packages.append(args[index])
            index += 1
    packages = [part for value in packages for part in value.split(",") if part]
    specs = packages or ([] if first is None else [first])
    return RunnerInvocation("npm", runner, tuple(dict.fromkeys([*specs, *maybe_packages])))


def _uv_invocation(runner: str, args: list[str]) -> RunnerInvocation:
    """``uvx`` / ``uv tool run``: the ``--from`` requirement (else the command), then every ``--with`` requirement.

    These options count only before the command; after it they are the
    server's arguments. ``--with`` (``-w``) takes one or more comma-separated
    requirements, and uv installs them next to the package, as it does the
    requirements in each ``--with-requirements`` file. Without a command uv
    only lists the installed tools, so it installs nothing.
    """
    options, command = _runner_options(args, _UVX_VALUE_FLAGS)
    package = _package_argument(options, command, "--from")
    if package is None:
        return RunnerInvocation("pypi", runner, ())
    extras = _with_requirements(options)
    return RunnerInvocation(
        "pypi",
        runner,
        (package, *extras),
        requirement_files=tuple(_flag_values(options, ("--with-requirements",))),
        with_specs=extras,
    )


def is_local_spec(spec: str) -> bool:
    """Whether a package spec names a local path (``./pkg``, ``/abs``, ``~/x``, ``file:``, ``${CLAUDE_PLUGIN_ROOT}/x``)."""
    return spec.startswith(_LOCAL_SPEC_PREFIXES)


def is_remote_npm_spec(spec: str) -> bool:
    """Whether an npm spec is fetched from git or a URL (``github:o/r``, ``git+https://...``, ``o/r``), not the registry."""
    return spec.startswith(_REMOTE_SPEC_PREFIXES) or (not spec.startswith("@") and "/" in spec)


def is_remote_pypi_spec(spec: str) -> bool:
    """Whether a PyPI spec is fetched from git or a URL (``git+https://...``, ``pkg @ git+https://...``)."""
    return spec.startswith(_REMOTE_SPEC_PREFIXES) or "@ git+" in spec or "@git+" in spec


def split_npm_spec(spec: str) -> tuple[str, str | None]:
    """``(name, version)`` of an npm registry spec; ``version`` is ``None`` when the spec names none.

    A scope's leading ``@`` belongs to the name: ``@scope/pkg@1.2.3`` is
    ``("@scope/pkg", "1.2.3")``.
    """
    at = spec.find("@", 1) if spec.startswith("@") else spec.find("@")
    if at <= 0:
        return spec, None
    return spec[:at], spec[at + 1 :]


def exact_npm_version(version: str) -> str | None:
    """The exact version an npm version spec names (``1.2.3``, ``=1.2.3``, ``v1.2.3``), else ``None``."""
    match = _NPM_EXACT_RE.match(version.strip())
    return match.group(1) if match else None


def is_pep440_version(text: str) -> bool:
    """Whether *text* is one valid PEP 440 version, in any spelling PEP 440 accepts."""
    return PEP440_VERSION_RE.fullmatch(text) is not None


@dataclass(frozen=True)
class PypiPin:
    """The version a PyPI requirement pins with ``==``, ``===``, or uv's ``@``.

    ``auditable`` is set when the version is a valid PEP 440 version, so the
    dependency audit can hand ``name==version`` to pip-audit. A ``===`` pin of
    any other string still pins the package, but no release can be matched to
    it, so the audit reports it unverified.
    """

    version: str
    auditable: bool


def specifier_pin(operator: str, version: str) -> PypiPin | None:
    """What one version specifier pins: ``==V`` and uv's ``@V`` when ``V`` is a PEP 440 version, and ``===V``.

    The MCP pinning check and the dependency audit (``requirements*.txt``,
    ``pyproject.toml``, and runner specs) all decide exactness here.
    """
    if operator == "===":
        return PypiPin(version, auditable=is_pep440_version(version))
    if operator in {"==", "@"} and is_pep440_version(version):
        return PypiPin(version, auditable=True)
    return None


def pypi_pin(requirement: str) -> PypiPin | None:
    """The version a PyPI runner requirement pins, else ``None``.

    Reads ``pkg==V``, ``pkg===V``, uv's ``pkg@V``, extras (``pkg[x]==V``), and
    the PEP 508 parenthesized form ``pkg (==V)``; an environment marker after
    ``;`` is ignored. Ranges, wildcards, tags such as ``@latest``, and URLs pin
    nothing (see :func:`specifier_pin`).
    """
    match = _PYPI_REQUIREMENT_RE.fullmatch(requirement.split(";", 1)[0].strip())
    if match is None:
        return None
    specifier = match.group("specifier").strip()
    if specifier.startswith("(") and specifier.endswith(")"):
        specifier = specifier[1:-1].strip()
    pin = _PIN_SPECIFIER_RE.fullmatch(specifier)
    return None if pin is None else specifier_pin(pin.group("operator"), pin.group("version"))


# --------------------------------------------------------------------------- #
# Pinning classification                                                      #
# --------------------------------------------------------------------------- #
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
    if is_local_spec(spec):
        return McpPinning("not_applicable", f"local package path {spec!r}")
    if spec.startswith("$"):
        return McpPinning("unpinned", f"package spec {spec!r} is taken from an environment reference")
    if is_remote_npm_spec(spec):
        if _GIT_SHA_RE.search(spec):
            return McpPinning("pinned", f"git/URL spec pinned to a commit: {spec!r}", remote=True)
        ref = _git_ref(spec)
        if ref and _is_floating_tag(ref, composite=False):
            return McpPinning(
                "unpinned",
                f"git/URL spec {spec!r} follows the moving branch or tag {ref!r}",
                floating=True,
                remote=True,
            )
        return McpPinning("unpinned", f"git/URL/GitHub spec without a commit SHA: {spec!r}", remote=True)
    _name, version = split_npm_spec(spec)
    if version is None:
        return McpPinning("unpinned", f"package {spec!r} has no version (resolves to the latest release)")
    if exact_npm_version(version):
        return McpPinning("pinned", f"exact version {spec!r}")
    if "$" in version:
        return McpPinning("unpinned", f"package {spec!r} takes its version from an environment reference")
    # A dist-tag starts with a letter; a range starts with a digit or an operator ('^1.0.0-beta' is a range).
    if _is_floating_tag(version, composite=version[:1].isalpha()):
        return McpPinning("unpinned", f"package {spec!r} uses the floating dist-tag {version!r}", floating=True)
    return McpPinning("unpinned", f"package {spec!r} uses a version range or dist-tag, not an exact version")


def _classify_python_spec(spec: str) -> McpPinning:
    """Classify a PyPI requirement spec as used by ``uvx`` / ``pipx run``."""
    if is_local_spec(spec):
        return McpPinning("not_applicable", f"local package path {spec!r}")
    if spec.startswith("$"):
        return McpPinning("unpinned", f"package spec {spec!r} is taken from an environment reference")
    if is_remote_pypi_spec(spec):
        if _GIT_SHA_RE.search(spec) or "#sha256=" in spec:
            return McpPinning("pinned", f"git/URL spec pinned to a commit or hash: {spec!r}", remote=True)
        ref = _git_ref(spec)
        if ref and _is_floating_tag(ref, composite=False):
            return McpPinning(
                "unpinned",
                f"git/URL spec {spec!r} follows the moving branch or tag {ref!r}",
                floating=True,
                remote=True,
            )
        return McpPinning("unpinned", f"git/URL spec without a commit SHA or hash: {spec!r}", remote=True)
    if pypi_pin(spec) is not None:
        return McpPinning("pinned", f"exact version {spec!r}")
    name, sep, version = spec.partition("@")
    if sep and name and _is_floating_tag(version, composite=False):
        # uv reads 'pkg@latest' as "the newest release, refreshed on every run".
        return McpPinning("unpinned", f"package {spec!r} uses the floating version {version.strip()!r}", floating=True)
    if any(marker in spec for marker in ("<", ">", "~", "!", "*", ",", "=", "@")):
        return McpPinning("unpinned", f"requirement {spec!r} is a range or tag, not an exact '==' version")
    return McpPinning("unpinned", f"package {spec!r} has no version (resolves to the latest release)")


def _classify_deno_module(module: str | None) -> McpPinning:
    """Classify the module ``deno run`` runs: an ``npm:`` or ``jsr:`` package, a remote module, or a local script."""
    if module and module.startswith(("npm:", "jsr:")):
        return _prefixed("deno run: ", _classify_npm_spec(module.split(":", 1)[1]))
    if module and module.startswith(("http://", "https://")):
        if _module_path_names_exact_version(module):
            return McpPinning("pinned", f"deno run: remote module pinned to an exact version: {module!r}", remote=True)
        return McpPinning("unpinned", f"deno run: remote module without an exact version: {module!r}", remote=True)
    return McpPinning("not_applicable", "deno run of a local script")


def _module_path_names_exact_version(url: str) -> bool:
    """Whether a remote module URL names an exact version in its path (``/x/mod@v1.2.3/mod.ts``).

    Only the path counts, the part the server reads: the query and fragment do
    not, and neither does a path with ``.`` or ``..`` segments (also
    percent-encoded), which the client resolves away.
    """
    try:
        path = urlsplit(whatwg_url(url)).path
    except ValueError:
        return False
    if any(unquote(segment) in {".", ".."} for segment in path.split("/")):
        return False
    return _DENO_EXACT_MODULE_PATH_RE.search(path) is not None


def _classify_go_module(spec: str | None) -> McpPinning:
    """``go run module/path@version``: pinned for an exact ``vX.Y.Z``; a local package is not applicable."""
    if spec is None or "@" not in spec:
        return McpPinning("not_applicable", "go run of a local package (versions come from go.mod)")
    version = spec.rpartition("@")[2]
    if _GO_EXACT_RE.match(version):
        return McpPinning("pinned", f"go run: exact module version {spec!r}")
    if _is_floating_tag(version, composite=False):
        return McpPinning("unpinned", f"go run: module {spec!r} uses the floating query {version!r}", floating=True)
    return McpPinning("unpinned", f"go run: module {spec!r} is not pinned to an exact vX.Y.Z version")


def _classify_nuget_spec(spec: str) -> McpPinning:
    """A .NET tool as ``dnx`` runs it: ``Package`` or ``Package@version``."""
    version = spec.rpartition("@")[2] if "@" in spec else ""
    if version and _NUGET_EXACT_RE.match(version):
        return McpPinning("pinned", f"exact version {spec!r}")
    if version and _is_floating_tag(version, composite=False):
        return McpPinning("unpinned", f"package {spec!r} uses the floating version {version!r}", floating=True)
    return McpPinning("unpinned", f"package {spec!r} has no exact version (resolves to the latest release)")


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


def _classify_spec_list(specs: Iterable[str], classify: Callable[[str], McpPinning]) -> McpPinning:
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


def _classify_python_invocation(invocation: RunnerInvocation) -> McpPinning:
    """``uvx``, ``uv tool run``, ``uv run``, and ``pipx run``: the tool spec and every ``--with`` requirement.

    ``--with`` installs more packages next to the tool, each resolved on every
    run unless exact, so the worst spec decides: a floating spec beats an
    unpinned one, which beats a pinned one, and the tool comes before its
    ``--with`` requirements. A ``--with-requirements`` file, whose packages are
    not read, leaves the launch unpinned.
    """
    runner, with_specs = invocation.runner, invocation.with_specs
    tools = invocation.specs[: len(invocation.specs) - len(with_specs)]
    pins = [(f"{runner}: ", _classify_python_spec(spec)) for spec in tools]
    pins += [(f"{runner} --with: ", _classify_python_spec(spec)) for spec in with_specs]
    for wanted in (lambda pin: pin.floating, lambda pin: pin.status == "unpinned"):
        found = next(((prefix, pin) for prefix, pin in pins if wanted(pin)), None)
        if found is not None:
            return _prefixed(*found)
    if invocation.requirement_files:
        shown = f"--with-requirements {invocation.requirement_files[0]}"
        return McpPinning("unpinned", f"{runner}: the packages of {shown!r} are not read, so they cannot be checked")
    if not pins:
        return McpPinning("unpinned", f"{runner} invocation without a package spec")
    return _prefixed(*next(((prefix, pin) for prefix, pin in pins if pin.status == "pinned"), pins[0]))


def _classify_invocation(invocation: RunnerInvocation) -> McpPinning:
    runner, specs = invocation.runner, invocation.specs
    spec = specs[0] if specs else None
    if invocation.ecosystem == "deno":
        return _classify_deno_module(spec)
    if invocation.ecosystem == "go":
        return _classify_go_module(spec)
    if invocation.ecosystem == "container":
        if spec is None:
            return McpPinning("unpinned", f"{runner} invocation without an image")
        return _prefixed(f"{runner}: ", _classify_image(spec))
    if invocation.ecosystem == "nuget":
        if spec is None:
            return McpPinning("unpinned", f"{runner} invocation without a package")
        return _prefixed(f"{runner}: ", _classify_nuget_spec(spec))
    if invocation.ecosystem == "pypi":
        return _classify_python_invocation(invocation)
    if not specs:
        # An npm runner without a package fetches nothing.
        return McpPinning("not_applicable", f"{runner} invocation without a package spec")
    return _prefixed(f"{runner}: ", _classify_spec_list(specs, _classify_npm_spec))


def _not_a_runner(argv: list[str]) -> McpPinning:
    """The classification of a launch that runs no package runner."""
    base, args = _command_basename(argv[0]), argv[1:]
    if base in _CONTAINER_RUNTIMES:
        return McpPinning("not_applicable", f"{base} invocation is not 'run'")
    if base == "uv":
        sub = _subcommand(args, _UV_RUN_VALUE_FLAGS)[0]
        if sub == "run":
            return McpPinning("not_applicable", "uv run of the project environment (no '--with' package)")
        if sub == "tool":
            return McpPinning("not_applicable", "uv tool invocation is not 'run'")
    return McpPinning("not_applicable", f"local interpreter, script, or binary ({base!r})")


def classify_mcp_pinning(config: Any, *, client: McpClient = "claude") -> McpPinning:
    """Classify whether one MCP declaration runs an exactly-pinned package.

    Package runners (``npx``, ``bunx``, ``bun x``, ``pnpm dlx``, ``yarn dlx``,
    ``npm exec``, ``uvx``, ``uv tool run``, ``uv run --with``, ``pipx run``,
    ``deno run`` of a registry spec, ``go run pkg@version``, ``dnx``, and
    ``docker|podman|nerdctl run``), read through :func:`parse_mcp_runner`, are
    ``pinned`` only when every package they install has an exact version
    (``pkg@1.2.3``, ``pkg==1.2.3``, ``--from pkg==1.2.3``, each ``--with``
    requirement, ``image:1.2.3``, ``image@sha256:...``); otherwise they are
    ``unpinned`` (``floating`` for a moving tag such as ``@latest``). Local
    interpreters and scripts (``node ./server.js``, ``python -m local_module``,
    ``./bin/server``), URL servers, and provider-only entries are
    ``not_applicable``.

    Claude Code expands ``${NAME:-default}`` in ``command`` and ``args`` (Cursor
    and Agent Plugins servers are read the same way, and so is a shell running a
    hook command), so the default the plugin ships is what gets classified:
    ``npx -y ${PKG:-@scope/mcp@latest}`` is floating. Codex runs the text as is.
    The detail is bounded and never carries a credential: neither one in the spec
    it quotes nor a credential flag's value that a runner reads as a package
    (``npx --token <value> pkg``).
    """
    launch = config if client == "codex" else _with_launch_defaults(config)
    pin = _classify_launch(launch)
    if launch is not config:
        pin = replace(pin, detail=f"{pin.detail}, with each ${{NAME:-default}} read at its default")
    return _redacted_pin(pin, _credential_values(config))


def _with_launch_defaults(config: Any) -> Any:
    """``config`` with every ``${NAME:-default}`` in ``command`` and ``args`` replaced by its default.

    The same object comes back when nothing has a default.
    """
    if not isinstance(config, dict) or not isinstance(config.get("command"), str):
        return config
    raw_args = config.get("args")
    fields = [config["command"], *(raw_args if isinstance(raw_args, list) else [])]
    if not any(isinstance(item, str) and _URL_EXPANSION_RE.search(item) for item in fields):
        return config
    expanded = [expand_url_defaults(item) if isinstance(item, str) else item for item in fields]
    if expanded == fields:
        return config
    launch = dict(config, command=expanded[0])
    if isinstance(raw_args, list):
        launch["args"] = expanded[1:]
    return launch


def _classify_launch(config: Any) -> McpPinning:
    if not isinstance(config, dict):
        return McpPinning("not_applicable", "declaration is not an object")
    argv = _argv(config)
    if argv is None:
        if isinstance(config.get("url"), str):
            return McpPinning("not_applicable", "remote url server (no package is installed)")
        return McpPinning("not_applicable", "provider-only or non-runnable declaration")
    invocation = _runner_invocation(argv)
    return _not_a_runner(argv) if invocation is None else _classify_invocation(invocation)


def mcp_container_image(config: Any) -> str | None:
    """Return the image reference a ``docker|podman|nerdctl run`` MCP server launches, if any.

    Read through :func:`parse_mcp_runner` (wrappers such as ``cmd /c`` are
    looked through), so a flag value is never mistaken for the image. Returns
    ``None`` for every other server kind.
    """
    invocation = parse_mcp_runner(config)
    if invocation is None or invocation.ecosystem != "container" or not invocation.specs:
        return None
    return invocation.specs[0]


def is_exact_container_image(image: str) -> bool:
    """True when an image reference names one immutable (digest) or exact-version (tag) image."""
    return _classify_image(image.strip()).status == "pinned"


def _prefixed(prefix: str, pin: McpPinning) -> McpPinning:
    return replace(pin, detail=f"{prefix}{pin.detail}")


def _credential_values(config: Any) -> list[str]:
    """The literal credentials in a declaration's ``command`` and ``args`` (see :func:`_inline_credentials`)."""
    if not isinstance(config, dict):
        return []
    command, args = config.get("command"), config.get("args")
    word_lists = (
        _split_command_line([command]) if isinstance(command, str) else [],
        [str(arg) for arg in args] if isinstance(args, list) else [],
    )
    values = {
        # '--token=<value>' holds its value after the '='; any other credential is the whole word.
        words[index].split("=", 1)[1] if flag is not None and words[index].startswith("-") else words[index]
        for words in word_lists
        for index, flag in _inline_credentials(words)
    }
    return sorted((value for value in values if value), key=len, reverse=True)


def _redacted_pin(pin: McpPinning, credentials: Iterable[str] = ()) -> McpPinning:
    """The classification with credentials in its detail removed, and the detail bounded.

    Each of ``credentials`` (a declaration's literal credentials) is withheld
    where the detail quotes it, and the rest is shown as :func:`_shown` shows
    plugin text, so a spec that is a URL with a token in it is redacted too.
    """
    detail = pin.detail
    for value in credentials:
        detail = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])", "<value withheld>", detail)
    return replace(pin, detail=_shown(detail, MAX_REPORT_CHARS))


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
IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network


@dataclass(frozen=True)
class EndpointClass:
    """A non-public MCP endpoint host found without any network access."""

    kind: EndpointKind
    reason: str
    host: str
    address: IPAddress | None = None
    encoded: bool = False
    # The host is this machine (a loopback name, 127.0.0.0/8, or ::1), so traffic to it stays local.
    is_loopback: bool = False


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
        return EndpointClass("private", "loopback", normalized, encoded=name_encoded, is_loopback=True)
    address, encoded = _parse_host_address(normalized)
    if address is None:
        return None
    found, embedded = _classify_address(address)
    if found is None:
        return None
    kind, reason = found
    # An embedded IPv4 address (mapped, 6to4, Teredo, NAT64, compatible) decides for the IPv6 literal.
    decided_by = _embedded_ipv4(address) if embedded else address
    return EndpointClass(
        kind,
        reason,
        normalized,
        address,
        encoded or name_encoded or embedded,
        is_loopback=decided_by is not None and decided_by.is_loopback,
    )


@dataclass(frozen=True)
class HostAllowlist:
    """Host policy entries (``mcp.allowed_private_hosts``, the hosts ``hooks.allowed_urls`` names), parsed once.

    Entries are host names, ``*.suffix`` wildcards, IP literals, or CIDR networks
    (e.g. ``10.0.0.0/8``), normalized the way WHATWG clients read a host. Cloud
    metadata endpoints are never allowed.
    """

    # Exact host names and IP literals, matched by equality.
    names: frozenset[str] = frozenset()
    # '.corp.example' for '*.corp.example': hosts below it match, the bare suffix does not.
    suffixes: tuple[str, ...] = ()
    # IP literal and CIDR entries, matched against a host's address (and an IPv4 address embedded in it).
    networks: tuple[IPNetwork, ...] = ()

    @classmethod
    def from_entries(cls, entries: Iterable[str]) -> HostAllowlist:
        names: set[str] = set()
        suffixes: list[str] = []
        networks: list[IPNetwork] = []
        for raw in entries:
            text = raw.strip().translate(_DOT_LOOKALIKES) if isinstance(raw, str) else ""
            if text.startswith("*."):
                # The suffix is normalized on its own: IDNA rejects the '*' label, which would
                # leave a Unicode suffix that never matches a host normalized to punycode.
                suffix = _normalize_host(text[2:])
                if suffix:
                    suffixes.append(f".{suffix}")
                continue
            entry = _normalize_host(text)
            if entry:
                names.add(entry)
                with contextlib.suppress(ValueError):  # a host name, not an IP literal or network
                    networks.append(ipaddress.ip_network(entry, strict=False))
        return cls(frozenset(names), tuple(dict.fromkeys(suffixes)), tuple(dict.fromkeys(networks)))

    @classmethod
    def of(cls, entries: HostAllowlist | Iterable[str]) -> HostAllowlist:
        """``entries`` itself when already parsed, else parsed from policy strings."""
        return entries if isinstance(entries, HostAllowlist) else cls.from_entries(entries)

    def allows(self, endpoint: EndpointClass) -> bool:
        """Whether an entry covers a non-public endpoint, by its host name or by a network holding its address."""
        if endpoint.kind == "metadata":
            return False
        if endpoint.host in self.names or endpoint.host.endswith(self.suffixes):
            return True
        candidates = [] if endpoint.address is None else [endpoint.address, _embedded_ipv4(endpoint.address)]
        return any(
            candidate is not None and candidate.version == network.version and candidate in network
            for candidate in candidates
            for network in self.networks
        )

    def allows_host(self, host: str, endpoint: EndpointClass | None) -> bool:
        """Whether an entry covers ``host``, whose static class is ``endpoint`` (``None`` when it looks public).

        A public-looking host matches by name only, so a public IP literal entry
        admits exactly that literal; a non-public host is checked by :meth:`allows`.
        """
        if endpoint is not None:
            return self.allows(endpoint)
        normalized = _normalize_host(host)
        return bool(normalized) and (normalized in self.names or normalized.endswith(self.suffixes))


def host_name_is_allowlisted(host: str, allowed_hosts: HostAllowlist | Iterable[str]) -> bool:
    """True when a policy entry names this host (exact name or ``*.suffix``).

    Only host names match here: IP literals and CIDR entries are checked against
    resolved addresses with :meth:`HostAllowlist.allows`, and cloud metadata host
    names are never allowlisted.
    """
    normalized = _normalize_host(host)
    if not normalized or _parse_host_address(normalized)[0] is not None:
        return False
    return HostAllowlist.of(allowed_hosts).allows_host(normalized, classify_endpoint_host(normalized))


def _check_endpoint(
    server: _ServerFindings, host: str | None, allowed_private_hosts: HostAllowlist, *, label: str, shown: str
) -> None:
    """Report a metadata or non-allowlisted private host of one URL field (``label``), shown as ``shown``."""
    if not host:
        return
    endpoint = classify_endpoint_host(host)
    if endpoint is None:
        return
    encoded = f" (encoded as {host!r})" if endpoint.encoded else ""
    if endpoint.kind == "metadata":
        server.report(
            Severity.HIGH,
            "mcp_endpoint_metadata",
            f"{label} targets a {endpoint.reason} endpoint{encoded}: {shown!r}; an MCP client pointed here "
            f"can expose instance credentials ({_ENDPOINT_STATIC_NOTE})",
            "Remove the instance-metadata endpoint; MCP servers must never target cloud metadata services.",
        )
        return
    if allowed_private_hosts.allows(endpoint):
        return
    server.report(
        Severity.MEDIUM,
        "mcp_endpoint_private",
        f"{label} host is a {endpoint.reason} address{encoded}: {shown!r}; the endpoint is not publicly "
        f"reachable and may target local services ({_ENDPOINT_STATIC_NOTE})",
        "Use a public HTTPS endpoint, or allow this intended private host through the validation policy "
        "(mcp.allowed_private_hosts).",
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


@dataclass(frozen=True)
class _OptionRisk:
    """What an agent-CLI option value does: the issue it raises, and whether only Codex reads it."""

    concept: Literal["permission_bypass_flag", "permission_mode_flag"]
    severity: Severity
    # '-a', '-s', and '-c' / '--config' are common options, so '-a never', '-s danger-full-access', and the
    # config overrides count only after a codex command in the same shell command or argv ('grep -a never f'
    # and 'tool -s danger-full-access' are not Codex).
    codex_only: bool = False


_BYPASS = _OptionRisk("permission_bypass_flag", Severity.HIGH)
_CODEX_BYPASS = _OptionRisk("permission_bypass_flag", Severity.HIGH, codex_only=True)
_PERMISSIVE_MODE = _OptionRisk("permission_mode_flag", Severity.MEDIUM)
# Agent-CLI options and the values that disable approvals or the sandbox (bypass) or let the launched agent
# approve some tool calls without a prompt (permissive mode): Claude Code's permission mode, Gemini CLI's approval
# mode, and Codex CLI's sandbox and approval policy (long and short option). Options and values match in any
# letter case and are reported as written here.
_OPTION_VALUE_RISKS: dict[str, dict[str, _OptionRisk]] = {
    "--permission-mode": {"bypassPermissions": _BYPASS, "acceptEdits": _PERMISSIVE_MODE, "auto": _PERMISSIVE_MODE},
    "--approval-mode": {"yolo": _BYPASS},
    "--sandbox": {"danger-full-access": _BYPASS},
    "-s": {"danger-full-access": _CODEX_BYPASS},
    "--ask-for-approval": {"never": _BYPASS},
    "-a": {"never": _CODEX_BYPASS},
}
# The same table by lower-case option and value, with the flag each value reports ("--permission-mode auto").
_OPTION_VALUES: dict[str, dict[str, tuple[str, _OptionRisk]]] = {
    option.lower(): {value.lower(): (f"{option} {value}", risk) for value, risk in values.items()}
    for option, values in _OPTION_VALUE_RISKS.items()
}
# Each option in one string: "--opt value", "--opt=value", or a quoted value.
_OPTION_VALUE_RES: tuple[tuple[re.Pattern[str], dict[str, tuple[str, _OptionRisk]]], ...] = tuple(
    (
        re.compile(
            rf"(?<![\w-]){re.escape(option)}(?:=|\s+)[\"']?(?P<value>{'|'.join(map(re.escape, values))})(?![\w-])",
            re.IGNORECASE,
        ),
        _OPTION_VALUES[option.lower()],
    )
    for option, values in _OPTION_VALUE_RISKS.items()
)
# Codex CLI config overrides with the bypass effect: '-c approval_policy=never',
# '--config sandbox_mode="danger-full-access"'.
PERMISSION_BYPASS_CONFIG: tuple[tuple[str, str], ...] = (
    ("approval_policy", "never"),
    ("sandbox_mode", "danger-full-access"),
)
# A codex command may be a wrapper named for it ('codex-wrapper', 'my-codex') or a variable ('${CODEX_BIN}').
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
# A config override as the argv value after '-c', and anywhere in one string.
_BYPASS_CONFIG_VALUE_RES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (f"-c {key}={value}", re.compile(rf"^[\"']?{key}\s*=\s*[\"']?{re.escape(value)}[\"']?$", re.IGNORECASE))
    for key, value in PERMISSION_BYPASS_CONFIG
)
_BYPASS_CONFIG_TEXT_RES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (
        f"-c {key}={value}",
        re.compile(
            rf"(?<![\w-])(?:-c|--config)(?:=|\s+)[\"']?{key}\s*=\s*[\"']?{re.escape(value)}(?![\w-])",
            re.IGNORECASE,
        ),
    )
    for key, value in PERMISSION_BYPASS_CONFIG
)
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


def _flag_hits(path: str, node: Any) -> Iterator[tuple[str, str, _OptionRisk]]:
    """Yield ``(json_path, flag, risk)`` for permission flags in a string or split across argv tokens."""
    if isinstance(node, str):
        node = _join_continuations(node)
        for match in _BYPASS_FLAG_RE.finditer(node):
            yield path, match.group(1).lower(), _BYPASS
        for pattern, values in _OPTION_VALUE_RES:
            for match in pattern.finditer(node):
                flag, risk = values[match.group("value").lower()]
                if not risk.codex_only or _codex_hit(node, match.start()):
                    yield path, flag, risk
        for flag, pattern in _BYPASS_CONFIG_TEXT_RES:
            if any(_codex_hit(node, match.start()) for match in pattern.finditer(node)):
                yield path, flag, _CODEX_BYPASS
        if any(_command_before(node, hit.start(), _GEMINI_COMMAND_RE) for hit in _GEMINI_YES_RE.finditer(node)):
            yield path, "gemini -y", _BYPASS
    elif isinstance(node, list):
        yield from _argv_flag_hits(path, node, codex=False)
    elif isinstance(node, dict):
        # {"command": "codex", "args": ["-a", "never"]}: the args follow a codex (or gemini) command.
        command, args = node.get("command"), node.get("args")
        if isinstance(command, str) and isinstance(args, list):
            args_path = f"{path}.args" if path else "args"
            if _CODEX_COMMAND_RE.search(command):
                yield from _argv_flag_hits(args_path, args, codex=True)
            if _GEMINI_COMMAND_RE.search(command):
                yield from _argv_flag_hits(args_path, args, codex=False, gemini=True)


def _argv_flag_hits(
    path: str, argv: list[Any], *, codex: bool, gemini: bool = False
) -> Iterator[tuple[str, str, _OptionRisk]]:
    """Options split across adjacent argv tokens (``["--sandbox", "danger-full-access"]``); the Codex-only
    forms count only after a codex token, or when ``codex`` says the argv belongs to one, and Gemini CLI's
    ``-y`` only after a gemini token (or when ``gemini`` says so)."""
    for index, token in enumerate(argv):
        if isinstance(token, str) and _GEMINI_COMMAND_RE.search(token):
            gemini = True
        elif gemini and isinstance(token, str) and token.strip() == "-y":
            yield f"{path}[{index}]", "gemini -y", _BYPASS
    for index, (option, value) in enumerate(itertools.pairwise(argv)):
        if isinstance(option, str) and _CODEX_COMMAND_RE.search(option):
            codex = True
        if not isinstance(option, str) or not isinstance(value, str):
            continue
        name = option.strip().lower()
        found = _OPTION_VALUES.get(name, {}).get(value.strip().strip("\"'").lower())
        if found is not None and (codex or not found[1].codex_only):
            flag, risk = found
            yield f"{path}[{index}]", flag, risk
        if codex and name in _BYPASS_CONFIG_OPTIONS:
            for flag, pattern in _BYPASS_CONFIG_VALUE_RES:
                if pattern.match(value.strip()):
                    yield f"{path}[{index}]", flag, _CODEX_BYPASS


def permission_flag_issues(value: Any) -> list[OverrideIssue]:
    """Find agent-CLI permission flags in any config/command string or argv list, in one bounded walk.

    Flags and options that disable approvals or the sandbox (``--dangerously-skip-permissions``,
    ``--yolo``, Gemini CLI's ``-y``, ``--permission-mode bypassPermissions``, ``--sandbox
    danger-full-access``, Codex's ``-a never``, ``-s danger-full-access``, and ``-c approval_policy=never``)
    are HIGH ``permission_bypass_flag`` issues; ``--permission-mode acceptEdits`` and ``auto`` are MEDIUM
    ``permission_mode_flag`` issues. Options and values match in any letter case. A config too large to
    walk completely adds a HIGH ``permission_bypass_scan_truncated`` issue last (fail closed).
    """
    issues: list[OverrideIssue] = []
    seen: set[tuple[str, str]] = set()
    walk = _ConfigWalk(value)
    for node_path, node in walk:
        for path, flag, risk in _flag_hits(node_path, node):
            if (path, flag) not in seen:
                seen.add((path, flag))
                issues.append(_permission_flag_issue(flag, path, risk))
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


def _permission_flag_issue(flag: str, path: str, risk: _OptionRisk) -> OverrideIssue:
    if risk.concept == "permission_mode_flag":
        return OverrideIssue(
            "permission_mode_flag",
            risk.severity,
            f"agent-CLI flag {flag!r}{f' in {path!r}' if path else ''} lets the launched agent approve some tool "
            "calls without a prompt",
            "Remove the flag; let the user choose the permission mode of any agent CLI the plugin launches.",
        )
    where = f" in '{path}'" if path else ""
    return OverrideIssue(
        "permission_bypass_flag",
        risk.severity,
        f"agent-CLI permission-bypass flag {flag!r}{where} disables tool-approval prompts or sandboxing",
        "Remove the permission-bypass flag; plugins must not disable the host agent's approvals or sandbox.",
    )


def permission_bypass_issues(value: Any) -> list[OverrideIssue]:
    """The HIGH part of :func:`permission_flag_issues`: permission-bypass flags, and a scan that stopped early."""
    return [issue for issue in permission_flag_issues(value) if issue.concept != "permission_mode_flag"]


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
        if looks_like_inline_secret(key, raw_value, query=False):
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


def _check_overrides(server: _ServerFindings, config: dict[str, Any]) -> None:
    """Agent-CLI permission flags anywhere in the entry (bypass flags, permissive modes such as ``--permission-mode
    acceptEdits``, and ``--allowedTools`` grants that run any command), dangerous env, and auto-approve keys."""
    from skillevaluator.plugin_component_risk import allowed_tools_flag_issues

    for issue in (
        *permission_flag_issues(config),
        *allowed_tools_flag_issues(config),
        *env_override_issues(config.get("env")),
        *auto_approve_issues(config),
    ):
        check = "mcp_auto_approve" if issue.concept == "auto_approve" else f"mcp_{issue.concept}"
        server.report(issue.severity, check, issue.message, issue.suggestion)


def _check_pinning(server: _ServerFindings, config: dict[str, Any], *, client: McpClient = "claude") -> None:
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
        server.report(
            Severity.HIGH,
            "mcp_command_floating_version",
            f"package runner uses a floating version or tag ({pin.detail}); each launch may fetch different code",
            "Pin the referenced package/image to an exact version or digest, not a moving tag such as latest or main.",
        )
        return
    server.report(
        Severity.MEDIUM,
        "mcp_unpinned_package",
        f"package runner is not pinned to an exact version ({pin.detail}); each launch may fetch different code",
        "Pin an exact version (pkg@1.2.3, pkg==1.2.3, --from pkg==1.2.3, image:1.2.3 or image@sha256:...).",
    )


def _check_cursor_references(server: _ServerFindings, config: dict[str, Any], client: McpClient) -> None:
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
    server.report(
        Severity.LOW,
        "mcp_env_reference_unverified",
        f"'${{env:NAME}}' in {where} is read as an environment reference, not an inline secret; {expands}",
        "Check that the client expands the reference; Claude Code expands ${NAME} and ${NAME:-default}.",
    )


def validate_mcp_server_declaration(
    name: Any,
    config: Any,
    file_path: str,
    *,
    allowed_private_hosts: HostAllowlist | Iterable[str] = (),
    manifest_type: str | None = None,
) -> list[Finding]:
    """Statically validate one contained ``mcpServers`` entry (``name`` -> config).

    ``allowed_private_hosts`` comes from the validation policy
    (``mcp.allowed_private_hosts``) and suppresses ``mcp_endpoint_private`` for
    intended private hosts; cloud metadata endpoints are never allowlisted. A
    caller that validates many servers can parse it once with
    :meth:`HostAllowlist.from_entries` and pass that. ``manifest_type`` names the
    format whose client loads the server (Claude Code by default); it decides
    how ``${VAR}`` references in the command and URL expand.
    """
    client = mcp_client_for_manifest(manifest_type)
    findings: list[Finding] = []

    if not isinstance(name, str) or not _MCP_NAME_RE.match(name.strip()):
        findings.append(
            mcp_finding(
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

    server = _ServerFindings(name, file_path, findings)
    if not isinstance(config, dict):
        server.report(
            Severity.HIGH,
            "mcp_config_not_object",
            "MCP server config must be a JSON object",
            "Express the MCP server config as an object with command/url/provider.",
        )
        return findings

    has_command = "command" in config
    has_url = "url" in config
    has_provider = "provider" in config
    declared_kinds = sum((has_command, has_url, has_provider))

    if declared_kinds == 0:
        server.report(
            Severity.HIGH,
            "mcp_missing_kind",
            "MCP server must declare a 'command' (stdio), a 'url' (http/sse), or a 'provider'",
            "Add a runnable command/url, or declare a public provider identifier.",
        )
    elif declared_kinds > 1:
        server.report(
            Severity.HIGH,
            "mcp_kind_invalid",
            "MCP server must declare exactly one of 'command', 'url', or 'provider'",
            "Choose one runnable or provider-only MCP form.",
        )

    _check_transport(server, config)
    _check_insecure_tls_config(server, config)
    _check_env_and_headers(server, config, client=client)
    _check_overrides(server, config)
    _check_cursor_references(server, config, client)

    if has_command:
        _check_command(server, config)
        _check_pinning(server, config, client=client)
    allowed = HostAllowlist.of(allowed_private_hosts)
    if has_url:
        _check_url(server, config, allowed, client=client)
    _check_oauth_urls(server, config, allowed, client=client)
    if has_provider and not (has_command or has_url):
        provider = config.get("provider")
        if not isinstance(provider, str) or not provider.strip():
            server.report(
                Severity.HIGH,
                "mcp_provider_invalid",
                "provider must be a non-empty string",
                "Set a public provider identifier.",
            )

    return findings
