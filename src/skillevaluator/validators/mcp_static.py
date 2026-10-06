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
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import unquote, urlparse

import idna

from skillevaluator.models.plugin import MCP_NAME_PATTERN
from skillevaluator.models.result import Finding, Severity
from skillevaluator.utils.structured_data import MAX_STRUCTURED_NODES
from skillevaluator.validators.url_policy import (
    has_secret_shape,
    is_credential_name,
    is_env_reference,
    looks_like_inline_secret,
    report_text,
    safe_url,
    url_ambiguities,
    url_credentials,
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
# Interpreters invoked with an inline program string execute arbitrary code.
_SHELL_INTERPRETERS: frozenset[str] = frozenset({"sh", "bash", "zsh", "dash", "ksh", "fish"})
# A shell's inline-program flag: '-c' alone or inside a short-option cluster
# ('-lc', '-ec', '-xc'); bash, sh and zsh also run '+c' as '-c'. Long options
# such as '--config' never match.
_SHELL_INLINE_PROGRAM_FLAG_RE = re.compile(r"[-+][A-Za-z]*c[A-Za-z]*")
# Shell options that take the next argument as their value: '--rcfile file',
# '--init-file file', and '-o name' / '-O name' (also '+o', '+O', and inside a
# cluster such as '-eo pipefail').
_SHELL_VALUE_LONG_OPTIONS: frozenset[str] = frozenset({"--rcfile", "--init-file"})
# Floating / non-pinned version markers (supply-chain drift risk).
_FLOATING_MARKERS: tuple[str, ...] = ("@latest", "@main", "@master", "@head", "@next", "@canary", ":latest", ":main")
# A marker counts only when it is attached to a package or image name ("pkg@latest",
# "img:main", "pkg@latest-beta"). An '@' that starts a token or follows a separator
# (space, '=', ',', ':', a slash, a quote) begins an npm scope such as
# "@nextcloud/..." or "@next-auth/...", and a marker that runs on into a longer
# word or host name ("@mainstay", "git@main.example.com") is not a tag.
_FLOATING_MARKER_RE = re.compile(
    r"(?<=[^\s=,:/\\'\"])(?:" + "|".join(map(re.escape, _FLOATING_MARKERS)) + r")(?![\w.])",
    re.IGNORECASE,
)

# Values that switch an environment or command-line setting on.
TRUTHY_VALUES: frozenset[str] = frozenset({"1", "true", "yes", "on"})
# Command flags that disable TLS/cert verification.
_INSECURE_TLS_FLAGS: frozenset[str] = frozenset(
    {"--insecure", "-k", "--no-check-certificate", "--tls-no-verify", "--ssl-no-verify", "--no-verify-tls"}
)


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


def _credential_flag_name(token: str) -> str | None:
    """Return the flag name when *token* is a credential-bearing option flag.

    Handles ``--api-key`` / ``--api-key=VALUE`` (and short ``-x`` / ``-x=VALUE``)
    forms. The flag name (leading dashes stripped) is matched against the same
    credential vocabulary used for env keys (:func:`~skillevaluator.validators.url_policy.is_credential_name`).
    """
    if not token.startswith("-"):
        return None
    flag = token.lstrip("-").split("=", 1)[0]
    return flag if flag and is_credential_name(flag) else None


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


def _check_url_inline_secrets(server: _ServerFindings, url: str, *, ambiguous: bool = False) -> None:
    """Flag credentials written into a URL's userinfo, query string, or fragment.

    The URL is read as written and, when it is ``ambiguous``, also the way WHATWG
    clients read it: they find userinfo that urllib does not see, for example in
    ``https:user:password@host``. A literal user name or password counts; a
    ``${VAR}`` reference does not, because the plugin then carries no secret.
    """
    readings = [url_credentials(url.strip(), userinfo_rule="literal")]
    if ambiguous:
        readings.append(url_credentials(whatwg_url(url), userinfo_rule="literal"))
    if any(reading.userinfo for reading in readings):
        server.report(
            Severity.CRITICAL,
            "mcp_url_inline_secret",
            f"url embeds inline userinfo credentials: {safe_url(url)!r} (userinfo withheld); only "
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
                f"url {part} parameter {shown!r} carries an inline credential; only ${{ENV}} references are allowed",
                "Do not put credentials in the URL query or fragment; reference a secret handle/env var instead.",
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
        return v in TRUTHY_VALUES
    return False


def _shown(text: str) -> str:
    """A command token or command line for messages: bounded and redacted, withheld whole when shaped like a secret."""
    return "<value withheld>" if has_secret_shape(text) else report_text(text, 120)


def _iter_command_tokens(config: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    command = config.get("command")
    if isinstance(command, str):
        tokens.append(command)
    args = config.get("args")
    if isinstance(args, list):
        tokens.extend(str(a) for a in args)
    return tokens


def validate_mcp_command(name: str, config: dict[str, Any], file_path: str, findings: list[Finding]) -> None:
    """Append the stdio command findings for one server's ``command`` and ``args`` to ``findings``.

    Shell metacharacters, a shell's inline program (``-c``), inline credentials,
    flags that disable TLS, and floating versions. Messages start with
    ``mcpServers['<name>']: ``.
    """
    _check_command(_ServerFindings(name, file_path, findings), config)


def validate_mcp_pinning(name: str, config: dict[str, Any], file_path: str, findings: list[Finding]) -> None:
    """Append a MEDIUM ``mcp_unpinned_package`` finding to ``findings`` when a package runner is not pinned.

    Not when ``findings`` already holds this server's blocking floating-version
    finding (from :func:`validate_mcp_command`), so a call after it never
    reports the same package twice.
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

    tokens = _iter_command_tokens(config)
    for token in tokens:
        if _SHELL_METACHAR_RE.search(token):
            server.report(
                Severity.CRITICAL,
                "mcp_command_shell_metacharacters",
                f"command token contains shell metacharacters: {_shown(token)!r}",
                "Remove shell operators (; | & ` $() < >). MCP commands run argv-style, not via a shell.",
            )
        if token in _INSECURE_TLS_FLAGS:
            server.report(
                Severity.CRITICAL,
                "mcp_command_disables_tls",
                f"command disables TLS/certificate verification: {_shown(token)!r}",
                "Remove insecure-TLS flags; do not disable certificate verification.",
            )
        if _FLOATING_MARKER_RE.search(token):
            server.report(
                Severity.HIGH,
                "mcp_command_floating_version",
                f"command token uses a floating (unpinned) version: {_shown(token)!r}",
                "Pin the referenced package/image to an exact version, not latest/main.",
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
            if value and not is_env_reference(value):
                server.report(
                    Severity.CRITICAL,
                    "mcp_command_inline_secret",
                    f"command argument {flag!r} carries an inline credential; only ${{ENV}} references are allowed",
                    'Pass the secret by reference (e.g. "${MY_TOKEN}"); never inline a raw credential in args.',
                )
                flagged_value_idx = value_idx
            continue
        if idx == flagged_value_idx:
            continue  # already reported as the preceding flag's value
        if has_secret_shape(token):
            server.report(
                Severity.CRITICAL,
                "mcp_command_inline_secret",
                f"command argument args[{idx}] contains an inline credential (value withheld)",
                'Pass the secret by reference (e.g. "${MY_TOKEN}"); never inline a raw credential in args.',
            )

    # Shell interpreter invoked with an inline program string (`sh -c "..."`, `bash -lc "..."`).
    shell = _shell_invocation(command, arg_list)
    if shell is not None and _shell_runs_inline_program(shell[1], shell=shell[0]):
        server.report(
            Severity.CRITICAL,
            "mcp_command_dangerous_form",
            f"command invokes a shell interpreter with '-c' ({_shown(command)!r}); this executes an arbitrary "
            "program string",
            "Invoke the server binary directly instead of wrapping it in a shell '-c' string.",
        )


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
# env options (GNU and BSD) whose value is the rest of the word or the next argument; '-S' /
# '--split-string' also splits its value into the arguments env reads next.
_ENV_VALUE_LETTERS = frozenset("uCPaLU")
_ENV_VALUE_LONG_OPTIONS: tuple[str, ...] = ("unset", "chdir", "argv0")
_ENV_ASSIGNMENT_RE = re.compile(r"[A-Za-z_]\w*=")


def _shell_invocation(command: str, args: list[str]) -> tuple[str, list[str]] | None:
    """``(shell, its arguments)`` when an MCP command runs a shell interpreter, else ``None``.

    The command is read like every other MCP command (:func:`_command_argv`). An
    ``env`` wrapper is looked through: its options, ``NAME=value`` assignments,
    and ``-S`` string (``env -i PATH=/bin bash -c ...``, ``/usr/bin/env -S "sh -c ..."``).
    """
    argv = _command_argv(command, args)
    while argv and _command_basename(argv[0]) == "env":
        argv = _env_command(argv[1:])
    if argv and _command_basename(argv[0]) in _SHELL_INTERPRETERS:
        return _command_basename(argv[0]), argv[1:]
    return None


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


def _shell_runs_inline_program(shell_args: list[str], *, shell: str = "sh") -> bool:
    """Return whether a shell's arguments select an inline program string ('-c').

    A shell reads options only until its first operand (the script it runs) or an
    end-of-options marker ('--' or '-'). Later arguments belong to the script, so
    a script's own '-config' or '-recursive' is not the shell's '-c'. fish also runs
    a program from '-C' / '--init-command' and spells '-c' as '--command'; its
    arguments are read both ways, so the fish reading can only flag more.
    """
    if _selects_inline_program(shell_args, _shell_option):
        return True
    return shell == "fish" and _selects_inline_program(shell_args, _fish_option)


def _selects_inline_program(shell_args: list[str], read_option: Callable[[str], tuple[bool, bool]]) -> bool:
    """Whether a shell's options, read one by one with ``read_option``, include an inline-program option."""
    index = 0
    while index < len(shell_args):
        arg = shell_args[index].strip()
        if arg in {"-", "--"} or not arg.startswith(("-", "+")):
            return False
        inline, takes_value = read_option(arg)
        if inline:
            return True
        index += 2 if takes_value else 1
    return False


def _shell_option(arg: str) -> tuple[bool, bool]:
    """``(runs an inline program, takes the next argument as its value)`` for one sh/bash/zsh/dash/ksh option."""
    if _SHELL_INLINE_PROGRAM_FLAG_RE.fullmatch(arg):
        return True, False
    if arg.startswith("--"):
        return False, arg in _SHELL_VALUE_LONG_OPTIONS
    return False, "o" in arg or "O" in arg


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


def _check_url(server: _ServerFindings, config: dict[str, Any], allowed_private_hosts: HostAllowlist) -> None:
    url = config.get("url")
    if not isinstance(url, str) or not url.strip():
        server.report(
            Severity.HIGH,
            "mcp_url_empty",
            "runnable MCP 'url' must be a non-empty string",
            "Set 'url' to the server endpoint using a secure https:// (or wss://) URL.",
        )
        return

    shown = safe_url(url)  # messages never echo userinfo or query credentials
    try:
        # ``raw`` is how urllib (and Python clients) read the text; ``parsed`` is how
        # WHATWG clients (Node and Rust MCP clients) read it. They differ only for an
        # ambiguous URL, which is flagged below.
        raw = urlparse(url.strip())
        parsed = urlparse(whatwg_url(url))
    except ValueError:  # e.g. an unbalanced '[' in the authority
        server.report(
            Severity.HIGH,
            "mcp_url_malformed_authority",
            "url could not be parsed (malformed authority)",
            "Use a valid host[:port] authority, e.g. https://host:443/path.",
        )
        return
    problems = url_ambiguities(url)
    if problems:
        server.report(
            Severity.HIGH,
            "mcp_url_malformed_authority",
            f"url contains {', and '.join(problems)}, so MCP clients and URL parsers disagree on where it "
            f"points; WHATWG clients (Node, the MCP SDKs) {_client_reading(url)}",
            "Write the URL with '//' after the scheme and without backslashes, whitespace, or control "
            "characters, e.g. https://host/path.",
        )
    scheme = (parsed.scheme or "").lower()
    # Inline credentials in userinfo/query are persisted verbatim; check them
    # independent of the scheme (secure https URLs are the common case).
    _check_url_inline_secrets(server, url, ambiguous=bool(problems))
    if problems:
        # A Python client may still connect where urllib reads the host: classify that one too.
        raw_host = _safe_hostname(raw)
        if raw_host and raw_host != _safe_hostname(parsed):
            _check_endpoint(server, url, raw_host, allowed_private_hosts)
    if scheme in ALLOWED_MCP_URL_SCHEMES:
        # A secure scheme alone is not a usable endpoint: require a host to connect
        # to, and reject a malformed authority/port. Otherwise a URL like "https://"
        # passes Tier 1 and only fails later in Harbor. Both are static, no network.
        try:
            host = parsed.hostname
            _ = parsed.port  # property access raises ValueError on a malformed port
        except ValueError:
            server.report(
                Severity.HIGH,
                "mcp_url_malformed_authority",
                f"url has a malformed authority/port: {shown!r}",
                "Use a valid host[:port] authority, e.g. https://host:443/path.",
            )
            return
        if not host:
            server.report(
                Severity.HIGH,
                "mcp_url_no_host",
                f"url uses scheme {scheme!r} but has no host to connect to: {shown!r}",
                "Provide a full endpoint with a hostname, e.g. https://host[:port]/path.",
            )
        _check_endpoint(server, url, host, allowed_private_hosts)
        return
    if scheme in _INSECURE_URL_SCHEMES:
        # Plaintext endpoints are blocked below; still report where they point.
        _check_endpoint(server, url, _safe_hostname(parsed), allowed_private_hosts)
    if scheme in _DANGEROUS_URL_SCHEMES or scheme == "":
        server.report(
            Severity.CRITICAL,
            "mcp_url_dangerous_scheme",
            f"url uses a dangerous/invalid scheme {scheme or '(none)'!r}: {shown!r}",
            "Use a secure https:// or wss:// endpoint; file/data/javascript/ftp schemes are not permitted.",
        )
    elif scheme in _INSECURE_URL_SCHEMES:
        server.report(
            Severity.HIGH,
            "mcp_url_insecure_scheme",
            f"url uses an insecure plaintext scheme {scheme!r}: {shown!r}",
            "Use https:// (or wss://) so the MCP transport is encrypted.",
        )
    else:
        server.report(
            Severity.HIGH,
            "mcp_url_scheme_not_allowed",
            f"url scheme {scheme!r} is not an allowed MCP scheme: {shown!r}",
            f"Use one of the allowed secure schemes: {', '.join(sorted(ALLOWED_MCP_URL_SCHEMES))}.",
        )


def _check_env_and_headers(server: _ServerFindings, config: dict[str, Any]) -> None:
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
        # NON-BLOCKING advisory: the evaluation runtime applies command+args (stdio)
        # and url (http/sse) only -- Harbor's per-MCP-server config has no env/headers
        # field and no agent adapter emits them, so this block will not reach the
        # launched MCP server (use task-level environment / CI credential injection
        # instead). The inline-secret / insecure-TLS checks below still run, so a raw
        # credential declared here is still caught and blocks.
        server.report(
            Severity.LOW,
            "mcp_field_ignored",
            f"'{section}' is not applied by the evaluation runtime and will be ignored; "
            "a Tier 3 run of this server is reported INCOMPLETE",
            f"Remove '{section}' or rely on task-level environment / CI credential injection; "
            "the runtime applies command+args (stdio) and url (http/sse) only.",
        )
        for key, value in block.items():
            if not isinstance(value, str):
                continue
            if _is_insecure_tls_env(key, value):
                server.report(
                    Severity.CRITICAL,
                    "mcp_insecure_tls_env",
                    f"'{section}.{key}' disables TLS/certificate verification",
                    "Do not disable TLS verification via environment variables.",
                )
            if looks_like_inline_secret(key, value):
                server.report(
                    Severity.CRITICAL,
                    "mcp_inline_secret",
                    f"'{section}.{key}' contains an inline credential; only ${{ENV}} references are allowed",
                    'Reference a secret handle/env var (e.g. "${MY_TOKEN}"); never inline a raw secret.',
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
RunnerEcosystem = Literal["npm", "pypi", "deno", "container"]

# The exact-version matchers are shared with the dependency audit, so a runner
# spec counts as pinned exactly when the audit can match it to one release.
# An exact npm version: "1.2.3", "=1.2.3", or "v1.2.3", with optional
# prerelease and build metadata.
_NPM_EXACT_RE = re.compile(r"^=?v?(\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?)$")
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
_VERSION_TAG_RE = re.compile(r"^v?\d+(?:\.\d+)*(?:[-+._][0-9A-Za-z.]+)*$")
# A remote module URL that names an exact version (``https://deno.land/x/mod@v1.2.3/mod.ts``).
_DENO_EXACT_MODULE_RE = re.compile(r"@v?\d+\.\d+\.\d+(?:[/?#]|$)")
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
# Every command _runner_invocation reads as a package runner.
_RUNNER_COMMANDS = frozenset(
    {"npx", "bunx", "pnpx", "pnpm", "yarn", "npm", "uvx", "uv", "pipx", "deno", *_CONTAINER_RUNTIMES}
)
# The programs whose name tells _command_argv that a 'command' with spaces is one program path.
_NAMED_PROGRAMS = frozenset({*_RUNNER_COMMANDS, *_SHELL_INTERPRETERS, "env"})


@dataclass(frozen=True)
class McpPinning:
    """Static supply-chain pinning classification of one MCP declaration.

    ``remote`` is set when the package comes from a git or URL spec, or a
    remote module, rather than from a registry.
    """

    status: PinStatus
    detail: str
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
    ``uv tool run``, ``docker run``). ``specs`` are read from the argv the way
    the runner reads it:

    * ``npm`` (``npx``, ``bunx``, ``pnpx``, ``pnpm dlx``, ``yarn dlx``,
      ``npm exec``): every ``-p``/``--package`` value, else the first
      positional argument;
    * ``pypi`` (``uvx``, ``uv tool run``, ``pipx run``): the ``--from`` or
      ``--spec`` requirement, else the first positional argument, then every
      ``uvx --with`` requirement;
    * ``deno`` (``deno run``): the module it runs, an ``npm:`` or ``jsr:``
      spec, a URL, or a local script;
    * ``container`` (``docker``, ``podman``, or ``nerdctl run``): the image.

    A runner's options end at the package, command, module, or image it runs
    (``npm exec`` reads them up to ``--``); later arguments belong to the
    server. ``specs`` is empty when the runner names no package or image.
    """

    ecosystem: RunnerEcosystem
    runner: str
    specs: tuple[str, ...]

    @property
    def npm_specs(self) -> tuple[str, ...]:
        """The npm registry specs the runner installs, including the package of ``deno run npm:pkg``."""
        if self.ecosystem == "npm":
            return self.specs
        if self.ecosystem == "deno":
            return tuple(spec.removeprefix("npm:") for spec in self.specs if spec.startswith("npm:"))
        return ()


def _command_basename(command: str) -> str:
    base = command.strip().replace("\\", "/").rsplit("/", 1)[-1].lower()
    for suffix in (".exe", ".cmd", ".bat", ".ps1"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _command_argv(command: str, args: list[str]) -> list[str]:
    """The argv that an MCP server's ``command`` and ``args`` run.

    ``command`` may name only the program, possibly as a path with spaces such as
    ``C:\\Program Files\\nodejs\\npx.cmd``, so every option is in ``args``; or it
    may hold a whole command line (``npx -y pkg``, ``bash -c node``), which is
    split into words. Its first word decides. The string is one program only when
    it starts like a path (its first word has a ``/`` or ``\\``) that names no
    runner, shell, or ``env``, while the whole string does. So ``npx -y pkg
    /srv/docker`` and ``bash -c x /usr/bin/env`` are command lines, whatever
    their last path segment names.
    """
    words = command.split()
    path_with_spaces = (
        ("/" in words[0] or "\\" in words[0])
        and _command_basename(words[0]) not in _NAMED_PROGRAMS
        and _command_basename(command) in _NAMED_PROGRAMS
    )
    return [command, *args] if path_with_spaces else [*words, *args]


def _argv(config: Any) -> list[str] | None:
    """The argv a runnable MCP declaration runs (:func:`_command_argv`), or ``None`` when it has no command."""
    if not isinstance(config, dict):
        return None
    command = config.get("command")
    if not isinstance(command, str) or not command.strip():
        return None
    raw_args = config.get("args")
    args = [str(arg) for arg in raw_args] if isinstance(raw_args, list) else []
    return _command_argv(command, args)


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


def _container_run_args(args: list[str]) -> list[str] | None:
    """The arguments after a container runtime's ``run`` or ``container run``; ``None`` for another subcommand."""
    if args[:1] == ["run"]:
        return args[1:]
    if args[:2] == ["container", "run"]:
        return args[2:]
    return None


def parse_mcp_runner(config: Any) -> RunnerInvocation | None:
    """The package runner an MCP declaration runs, or ``None`` when it runs none.

    This is the one argv reader for package runners: the pinning check
    (:func:`classify_mcp_pinning`), the container-image lookup
    (:func:`mcp_container_image`), and the dependency audit all read runner
    invocations through it, so they agree on what a server installs. Local
    interpreters, scripts, and binaries, container subcommands other than
    ``run``, URL servers, and provider-only entries give ``None``.
    """
    argv = _argv(config)
    return None if argv is None else _runner_invocation(argv)


def _runner_invocation(argv: list[str]) -> RunnerInvocation | None:
    base, args = _command_basename(argv[0]), argv[1:]
    if base in {"npx", "bunx", "pnpx"}:
        return _npm_invocation(base, args, _NPX_VALUE_FLAGS if base == "npx" else _DLX_VALUE_FLAGS)
    if base in {"pnpm", "yarn"} and args[:1] == ["dlx"]:
        return _npm_invocation(f"{base} dlx", args[1:], _DLX_VALUE_FLAGS)
    if base == "npm" and args[:1] in (["exec"], ["x"]):
        # Like every npm command, npm exec reads its options anywhere before "--".
        return _npm_invocation("npm exec", args[1:], _NPX_VALUE_FLAGS, options_until_separator=True)
    if base == "uvx":
        return _uv_invocation("uvx", args)
    if base == "uv" and args[:2] in (["tool", "run"], ["tool", "x"]):
        return _uv_invocation("uv tool run", args[2:])
    if base == "pipx" and args[:1] == ["run"]:
        options, app = _runner_options(args[1:], _PIPX_VALUE_FLAGS)
        spec = _package_argument(options, app, "--spec")
        return RunnerInvocation("pypi", "pipx run", () if spec is None else (spec,))
    if base == "deno" and args[:1] == ["run"]:
        module = _first_positional(args[1:], _DENO_VALUE_FLAGS)
        return RunnerInvocation("deno", "deno run", () if module is None else (module,))
    run_args = _container_run_args(args) if base in _CONTAINER_RUNTIMES else None
    if run_args is not None:
        image = _first_positional(run_args, _DOCKER_VALUE_FLAGS)
        return RunnerInvocation("container", f"{base} run", () if image is None else (image,))
    return None


def _npm_invocation(
    runner: str, args: list[str], value_flags: frozenset[str], *, options_until_separator: bool = False
) -> RunnerInvocation:
    """An npm package runner: every ``-p``/``--package`` value, else the first positional argument.

    The runner's options end at the package it runs, or, with
    *options_until_separator*, at ``--``.
    """
    if options_until_separator:
        options, spec = args, _first_positional(args, value_flags)
    else:
        options, spec = _runner_options(args, value_flags)
    packages = _flag_values(options, ("-p", "--package"))
    if packages:
        return RunnerInvocation("npm", runner, tuple(packages))
    return RunnerInvocation("npm", runner, () if spec is None else (spec,))


def _uv_invocation(runner: str, args: list[str]) -> RunnerInvocation:
    """``uvx`` / ``uv tool run``: the ``--from`` requirement (else the command), then every ``--with`` requirement.

    Both options count only before the command; after it they are the
    server's arguments. ``--with`` takes one or more comma-separated
    requirements, and uv installs them next to the package. Without a command
    uv only lists the installed tools, so it installs nothing.
    """
    options, command = _runner_options(args, _UVX_VALUE_FLAGS)
    package = _package_argument(options, command, "--from")
    if package is None:
        return RunnerInvocation("pypi", runner, ())
    extras = (item.strip() for value in _flag_values(options, ("--with",)) for item in value.split(","))
    return RunnerInvocation("pypi", runner, (package, *(item for item in extras if item)))


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


def _classify_npm_spec(spec: str) -> McpPinning:
    """Classify an npm package spec (``pkg``, ``@scope/pkg@1.2.3``, git/URL, local path)."""
    if is_local_spec(spec):
        return McpPinning("not_applicable", f"local package path {spec!r}")
    if is_remote_npm_spec(spec):
        if _GIT_SHA_RE.search(spec):
            return McpPinning("pinned", f"git/URL spec pinned to a commit: {spec!r}", remote=True)
        return McpPinning("unpinned", f"git/URL/GitHub spec without a commit SHA: {spec!r}", remote=True)
    _name, version = split_npm_spec(spec)
    if version is None:
        return McpPinning("unpinned", f"package {spec!r} has no version (resolves to the latest release)")
    if exact_npm_version(version):
        return McpPinning("pinned", f"exact version {spec!r}")
    return McpPinning("unpinned", f"package {spec!r} uses a version range or dist-tag, not an exact version")


def _classify_python_spec(spec: str) -> McpPinning:
    """Classify a PyPI requirement spec as used by ``uvx`` / ``pipx run``."""
    if is_local_spec(spec):
        return McpPinning("not_applicable", f"local package path {spec!r}")
    if is_remote_pypi_spec(spec):
        if _GIT_SHA_RE.search(spec) or "#sha256=" in spec:
            return McpPinning("pinned", f"git/URL spec pinned to a commit or hash: {spec!r}", remote=True)
        return McpPinning("unpinned", f"git/URL spec without a commit SHA or hash: {spec!r}", remote=True)
    if pypi_pin(spec) is not None:
        return McpPinning("pinned", f"exact version {spec!r}")
    if any(marker in spec for marker in ("<", ">", "~", "!", "*", ",", "=", "@")):
        return McpPinning("unpinned", f"requirement {spec!r} is a range or tag, not an exact '==' version")
    return McpPinning("unpinned", f"package {spec!r} has no version (resolves to the latest release)")


def _classify_deno_module(module: str | None) -> McpPinning:
    """Classify the module ``deno run`` runs: an ``npm:`` or ``jsr:`` package, a remote module, or a local script."""
    if module and module.startswith(("npm:", "jsr:")):
        return _prefixed("deno run: ", _classify_npm_spec(module.split(":", 1)[1]))
    if module and module.startswith(("http://", "https://")):
        if _DENO_EXACT_MODULE_RE.search(module):
            return McpPinning("pinned", f"deno run: remote module pinned to an exact version: {module!r}", remote=True)
        return McpPinning("unpinned", f"deno run: remote module without an exact version: {module!r}", remote=True)
    return McpPinning("not_applicable", "deno run of a local script")


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


def _classify_spec_list(specs: Iterable[str], classify: Any) -> McpPinning:
    """Classify several specs as one: unpinned when any spec is, else pinned when any is."""
    results = [classify(spec) for spec in specs]
    unpinned = [result for result in results if result.status == "unpinned"]
    if unpinned:
        return unpinned[0]
    pinned = [result for result in results if result.status == "pinned"]
    if pinned:
        return pinned[0]
    return results[0]


def _prefixed(prefix: str, pin: McpPinning) -> McpPinning:
    return McpPinning(pin.status, f"{prefix}{pin.detail}", pin.remote)


def _classify_invocation(invocation: RunnerInvocation) -> McpPinning:
    runner, specs = invocation.runner, invocation.specs
    if invocation.ecosystem == "deno":
        return _classify_deno_module(specs[0] if specs else None)
    if invocation.ecosystem == "container":
        if not specs:
            return McpPinning("unpinned", f"{runner} invocation without an image")
        return _prefixed(f"{runner}: ", _classify_image(specs[0]))
    if not specs:
        # An npm runner without a package fetches nothing; uvx and pipx run need one.
        status: PinStatus = "not_applicable" if invocation.ecosystem == "npm" else "unpinned"
        return McpPinning(status, f"{runner} invocation without a package spec")
    classify = _classify_npm_spec if invocation.ecosystem == "npm" else _classify_python_spec
    return _prefixed(f"{runner}: ", _classify_spec_list(specs, classify))


def classify_mcp_pinning(config: Any) -> McpPinning:
    """Classify whether one MCP declaration runs an exactly-pinned package.

    Package runners (``npx``, ``bunx``, ``pnpm dlx``, ``yarn dlx``, ``npm exec``,
    ``uvx``, ``uv tool run``, ``pipx run``, ``deno run`` of a registry spec, and
    ``docker|podman run``), read through :func:`parse_mcp_runner`, are
    ``pinned`` only when every package they install has an exact version
    (``pkg@1.2.3``, ``pkg==1.2.3``, ``--from pkg==1.2.3``, each ``uvx --with``
    requirement, ``image:1.2.3``, ``image@sha256:...``); otherwise they are
    ``unpinned``. Local interpreters and scripts (``node ./server.js``,
    ``python -m local_module``, ``./bin/server``), URL servers, and
    provider-only entries are ``not_applicable``.
    """
    if not isinstance(config, dict):
        return McpPinning("not_applicable", "declaration is not an object")
    argv = _argv(config)
    if argv is None:
        if isinstance(config.get("url"), str):
            return McpPinning("not_applicable", "remote url server (no package is installed)")
        return McpPinning("not_applicable", "provider-only or non-runnable declaration")
    invocation = _runner_invocation(argv)
    if invocation is not None:
        return _classify_invocation(invocation)
    base = _command_basename(argv[0])
    if base in _CONTAINER_RUNTIMES:
        return McpPinning("not_applicable", f"{base} invocation is not 'run'")
    return McpPinning("not_applicable", f"local interpreter, script, or binary ({base!r})")


def mcp_container_image(config: Any) -> str | None:
    """Return the image reference a ``docker|podman|nerdctl run`` MCP server launches, if any.

    Read through :func:`parse_mcp_runner`, so a flag value is never mistaken
    for the image. Returns ``None`` for every other server kind.
    """
    invocation = parse_mcp_runner(config)
    if invocation is None or invocation.ecosystem != "container" or not invocation.specs:
        return None
    return invocation.specs[0]


def is_exact_container_image(image: str) -> bool:
    """True when an image reference names one immutable (digest) or exact-version (tag) image."""
    return _classify_image(image.strip()).status == "pinned"


# --------------------------------------------------------------------------- #
# Network-free endpoint policy                                                #
# --------------------------------------------------------------------------- #
EndpointKind = Literal["metadata", "private"]

_METADATA_HOSTNAMES = frozenset({"metadata.google.internal", "metadata"})
_METADATA_ADDRESSES = frozenset(
    {
        ipaddress.ip_address("169.254.169.254"),  # AWS, Azure, GCP, OpenStack, and most other clouds
        ipaddress.ip_address("fd00:ec2::254"),  # AWS IMDS over IPv6
        ipaddress.ip_address("100.100.100.200"),  # Alibaba Cloud
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
            entry = _normalize_host(raw) if isinstance(raw, str) else ""
            if entry.startswith("*."):
                suffixes.append(entry[1:])
            elif entry:
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


def host_is_allowlisted(endpoint: EndpointClass, allowed_hosts: HostAllowlist | Iterable[str]) -> bool:
    """True when a policy entry allows this private host.

    Entries are exact host names, ``*.suffix`` wildcards, IP literals, or CIDR
    networks (e.g. ``10.0.0.0/8``). Cloud metadata endpoints are never allowlisted.
    """
    return HostAllowlist.of(allowed_hosts).allows(endpoint)


def host_name_is_allowlisted(host: str, allowed_hosts: HostAllowlist | Iterable[str]) -> bool:
    """True when a policy entry names this host (exact name or ``*.suffix``).

    Only host names match here: IP literals and CIDR entries are checked against
    resolved addresses with :func:`host_is_allowlisted`, and cloud metadata host
    names are never allowlisted.
    """
    normalized = _normalize_host(host)
    if not normalized or _parse_host_address(normalized)[0] is not None:
        return False
    return HostAllowlist.of(allowed_hosts).allows_host(normalized, classify_endpoint_host(normalized))


def _check_endpoint(server: _ServerFindings, url: str, host: str | None, allowed_private_hosts: HostAllowlist) -> None:
    if not host:
        return
    endpoint = classify_endpoint_host(host)
    if endpoint is None:
        return
    encoded = f" (encoded as {host!r})" if endpoint.encoded else ""
    shown = safe_url(url)  # never echo userinfo or query credentials
    if endpoint.kind == "metadata":
        server.report(
            Severity.HIGH,
            "mcp_endpoint_metadata",
            f"url targets a {endpoint.reason} endpoint{encoded}: {shown!r}; an MCP client pointed here "
            f"can expose instance credentials ({_ENDPOINT_STATIC_NOTE})",
            "Remove the instance-metadata endpoint; MCP servers must never target cloud metadata services.",
        )
        return
    if allowed_private_hosts.allows(endpoint):
        return
    server.report(
        Severity.MEDIUM,
        "mcp_endpoint_private",
        f"url host is a {endpoint.reason} address{encoded}: {shown!r}; the endpoint is not publicly "
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
    # '-a' is a common short option and '-c' / '--config' a common flag, so '-a never' and the config
    # overrides count only after a codex command in the same string or argv ('grep -a never f' is not Codex).
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
    "-s": {"danger-full-access": _BYPASS},
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
_CODEX_COMMAND_RE = re.compile(r"(?<![\w.-])codex(?:\.exe|\.cmd)?(?![\w.-])|\$\{?CODEX\w*", re.IGNORECASE)
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


def _codex_hit(text: str, start: int) -> bool:
    """Whether a codex command comes before ``start`` in ``text``."""
    return _CODEX_COMMAND_RE.search(text, 0, start) is not None


def _flag_hits(path: str, node: Any) -> Iterator[tuple[str, str, _OptionRisk]]:
    """Yield ``(json_path, flag, risk)`` for permission flags in a string or split across argv tokens."""
    if isinstance(node, str):
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
    elif isinstance(node, list):
        yield from _argv_flag_hits(path, node, codex=False)
    elif isinstance(node, dict):
        # {"command": "codex", "args": ["-a", "never"]}: the args follow a codex command.
        command, args = node.get("command"), node.get("args")
        if isinstance(command, str) and isinstance(args, list) and _CODEX_COMMAND_RE.search(command):
            yield from _argv_flag_hits(f"{path}.args" if path else "args", args, codex=True)


def _argv_flag_hits(path: str, argv: list[Any], *, codex: bool) -> Iterator[tuple[str, str, _OptionRisk]]:
    """Options split across adjacent argv tokens (``["--sandbox", "danger-full-access"]``); the Codex-only
    forms count only after a codex token, or when ``codex`` says the argv belongs to one."""
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
    ``--yolo``, ``--permission-mode bypassPermissions``, ``--sandbox danger-full-access``, Codex's
    ``-a never`` and ``-c approval_policy=never``) are HIGH ``permission_bypass_flag`` issues;
    ``--permission-mode acceptEdits`` and ``auto`` are MEDIUM ``permission_mode_flag`` issues. Options
    and values match in any letter case. A config too large to walk completely adds a HIGH
    ``permission_bypass_scan_truncated`` issue last (fail closed).
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


def _check_overrides(server: _ServerFindings, config: dict[str, Any]) -> None:
    """Permission-bypass flags anywhere in the entry, dangerous env, and auto-approve keys."""
    for issue in (
        *permission_bypass_issues(config),
        *env_override_issues(config.get("env")),
        *auto_approve_issues(config),
    ):
        check = "mcp_auto_approve" if issue.concept == "auto_approve" else f"mcp_{issue.concept}"
        server.report(issue.severity, check, issue.message, issue.suggestion)


def _check_pinning(server: _ServerFindings, config: dict[str, Any]) -> None:
    pin = classify_mcp_pinning(config)
    if pin.status != "unpinned":
        return
    # A floating marker (@latest, :latest, ...) already raised the blocking
    # mcp_command_floating_version finding for this entry; do not double-report.
    if any(
        f.check_name == "mcp_command_floating_version" and f.metadata.get("mcp_server") == server.name
        for f in server.findings
    ):
        return
    server.report(
        Severity.MEDIUM,
        "mcp_unpinned_package",
        f"package runner is not pinned to an exact version ({pin.detail}); each launch may fetch different code",
        "Pin an exact version (pkg@1.2.3, pkg==1.2.3, --from pkg==1.2.3, image:1.2.3 or image@sha256:...).",
    )


def validate_mcp_server_declaration(
    name: Any,
    config: Any,
    file_path: str,
    *,
    allowed_private_hosts: HostAllowlist | Iterable[str] = (),
) -> list[Finding]:
    """Statically validate one contained ``mcpServers`` entry (``name`` -> config).

    ``allowed_private_hosts`` comes from the validation policy
    (``mcp.allowed_private_hosts``) and suppresses ``mcp_endpoint_private`` for
    intended private hosts; cloud metadata endpoints are never allowlisted. A
    caller that validates many servers can parse it once with
    :meth:`HostAllowlist.from_entries` and pass that.
    """
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
    _check_env_and_headers(server, config)
    _check_overrides(server, config)

    if has_command:
        _check_command(server, config)
        _check_pinning(server, config)
    if has_url:
        _check_url(server, config, HostAllowlist.of(allowed_private_hosts))
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
