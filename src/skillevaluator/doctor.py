# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 1 and Tier 2 readiness checks for ``doctor`` and ``health-check``.

Tier 3 readiness lives in :func:`skillevaluator.tier3.commands.doctor`. This
module never imports Harbor, and Tier 2 loads provider SDKs only for an
explicit live probe, so Tier 1 readiness works on a base install.
"""

from __future__ import annotations

import importlib.util
import math
import re
from collections.abc import Callable
from typing import Any

from rich.box import SIMPLE
from rich.console import Console
from rich.table import Table

from skillevaluator import __version__
from skillevaluator.utils.rich_markup import escape_markup

Row = tuple[str, str, str]

DEFAULT_PROBE_TIMEOUT_SEC = 60.0

# Tier 1 checks that run an external scanner, with the ``Tools`` entries they
# launch, in the order the checks run them.
_SCANNERS_BY_CHECK: dict[str, tuple[str, ...]] = {
    "security": ("skillspector",),
    "code-integrity": ("bandit", "semgrep", "gitleaks"),
    "dependency": ("pip_audit",),
}
# Older releases lack the CLI or report contract the validators parse: the
# Bandit floor matches the security extra, and Gitleaks 8 introduced the
# ``detect`` command the secrets scan runs.
_MINIMUM_SCANNER_VERSIONS: dict[str, tuple[int, int, int]] = {
    "bandit": (1, 9, 4),
    "gitleaks": (8, 0, 0),
}
_SCANNER_LABELS = {"skillspector": "SkillSpector"}
_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")
_PROBE_TEXT = "SkillEvaluator readiness check"

# emoji=False: ":name:" codes in configuration values must print literally.
console = Console(emoji=False)


def _probe_run_options(command: str) -> dict[str, Any]:
    """Launch a version probe with the environment the scanner's validator uses."""
    if command == "skillspector":
        # The static SkillSpector scan runs without provider credentials.
        from skillevaluator.validators.security import _skillspector_process_env

        return {"env": _skillspector_process_env(), "replace_env": True}
    if command == "semgrep":
        # The scan passes --disable-version-check; keep the probe offline too.
        return {"env": {"SEMGREP_ENABLE_VERSION_CHECK": "0"}}
    return {}


def _scanner_row(tool: Any, timeout: float) -> Row:
    """Report one scanner's resolved executable and version without a shell."""
    label = _SCANNER_LABELS.get(tool.command, tool.name)
    if tool.configuration_error:
        return (label, "fail", tool.configuration_error)
    if not tool.is_available:
        return (label, "fail", f"not installed. {tool.get_install_hint()}")
    location = tool.path
    if tool.resolution_source != "automatic":
        location += f" (from {tool.override_env})"
    version_args = ["version"] if tool.command == "gitleaks" else ["--version"]
    result = tool.run(version_args, timeout=timeout, log_command=False, **_probe_run_options(tool.command))
    if result.error_message:
        return (label, "fail", result.error_message)
    if result.exit_code != 0:
        return (label, "fail", f"version probe exited with status {result.exit_code}; {location}")
    match = _VERSION_RE.search(result.stdout) or _VERSION_RE.search(result.stderr)
    if match is None:
        return (label, "warn", f"version not reported; {location}")
    version = tuple(int(part) for part in match.groups())
    minimum = _MINIMUM_SCANNER_VERSIONS.get(tool.command)
    if minimum is not None and version < minimum:
        required = ".".join(str(part) for part in minimum)
        return (
            label,
            "fail",
            f"{match.group(0)} at {location} is older than the supported {required}. {tool.get_install_hint()}",
        )
    return (label, "pass", f"{match.group(0)} at {location}")


def _tier1_rows(checks: str | None, timeout: float) -> list[Row]:
    from skillevaluator.tier1.commands import RECOGNIZED_CHECKS, enabled_check_lineup
    from skillevaluator.utils.tool_runner import Tools

    lineup = enabled_check_lineup(checks)
    unknown = sorted(set(lineup) - RECOGNIZED_CHECKS)
    if unknown:
        raise ValueError(f"Unknown Tier 1 check(s): {', '.join(unknown)}")
    names = dict.fromkeys(name for check in lineup for name in _SCANNERS_BY_CHECK.get(check, ()))
    if not names:
        return [("Tier 1 scanners", "pass", "the selected checks run no external scanner")]
    return [_scanner_row(getattr(Tools, name), timeout) for name in names]


def _module_available(module: str) -> bool:
    # find_spec does not import the module, and may raise when a meta-path
    # blocker is active; either way the extra is unavailable.
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _live_row(name: str, probe: Callable[[], str], diagnose: Callable[[Exception], str]) -> Row:
    """Run one live request; report only a bounded diagnostic, never a provider body."""
    try:
        detail = probe()
    except Exception as exc:
        return (name, "fail", diagnose(exc))
    return (name, "pass", detail)


def _probe_embedding(http_client: Any) -> str:
    from skillevaluator.embedding.client import EmbeddingClient

    vector = EmbeddingClient(max_retries=0, http_client=http_client).embed_single(_PROBE_TEXT)
    return f"returned a {len(vector)}-dimension vector"


def _embedding_failure(exc: Exception) -> str:
    from skillevaluator.embedding.client import SimilarityConfigError

    if isinstance(exc, SimilarityConfigError):
        # EmbeddingClient builds these messages without provider response bodies.
        return str(exc)
    return f"Embedding request failed ({type(exc).__name__})."


def _probe_chat(http_client: Any) -> str:
    from skillevaluator.constants import CONTENT_DEDUP_LLM_REASONING_EFFORT
    from skillevaluator.inference.client import LLMClient

    # The same client settings as Tier 2 duplicate analysis, without retries.
    client = LLMClient(
        reasoning_effort=CONTENT_DEDUP_LLM_REASONING_EFFORT,
        max_retries=0,
        http_client=http_client,
    )
    client.completions("Reply with the single word OK.", _PROBE_TEXT)
    return "returned a chat completion"


def _tier2_rows(*, verify_models: bool, timeout: float) -> list[Row]:
    if not _module_available("openai"):
        return [("Tier 2 extra", "fail", "not installed. Reinstall SkillEvaluator with the tier2 or all extra.")]

    from skillevaluator.inference.diagnostics import llm_failure_diagnostic, safe_llm_labels
    from skillevaluator.provider_config import (
        ProviderConfigurationError,
        resolve_embedding_provider,
        resolve_llm_provider,
    )

    rows: list[Row] = []
    resolved: set[str] = set()
    for label, resolve in (("Embedding provider", resolve_embedding_provider), ("LLM provider", resolve_llm_provider)):
        try:
            config = resolve()
        except ProviderConfigurationError as exc:
            rows.append((label, "fail", str(exc)))
            continue
        provider, model = safe_llm_labels(config.provider, config.model)
        rows.append((label, "pass", f"{provider} / {model}"))
        resolved.add(label)
    if not verify_models:
        return rows

    import httpx

    # One shared client bounds every request; the SDK clients never retry.
    with httpx.Client(timeout=timeout) as http_client:
        if "Embedding provider" in resolved:
            rows.append(_live_row("Embedding model", lambda: _probe_embedding(http_client), _embedding_failure))
        if "LLM provider" in resolved:
            rows.append(_live_row("Deduplication model", lambda: _probe_chat(http_client), llm_failure_diagnostic))
    return rows


def readiness_rows(
    *,
    tier: str,
    checks: str | None = None,
    verify_models: bool = False,
    timeout: float = DEFAULT_PROBE_TIMEOUT_SEC,
) -> list[Row]:
    """Return ``(check, pass|warn|fail, detail)`` rows for Tier 1 or Tier 2.

    Tier 1 reports the external scanner each selected check runs. Tier 2
    resolves the embedding and LLM providers offline; ``verify_models`` also
    sends one embedding and one chat request. ``timeout`` bounds each scanner
    probe and each live request.
    """
    if tier not in {"1", "2"}:
        raise ValueError("Expected tier 1 or 2")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be finite and positive")
    rows: list[Row] = [("CLI package", "pass", f"skillevaluator {__version__}")]
    if tier == "1":
        rows.extend(_tier1_rows(checks, timeout))
    else:
        rows.extend(_tier2_rows(verify_models=verify_models, timeout=timeout))
    return rows


def doctor(
    *,
    tier: str,
    checks: str | None = None,
    verify_models: bool = False,
    timeout: float = DEFAULT_PROBE_TIMEOUT_SEC,
) -> int:
    """Print Tier 1 or Tier 2 readiness and return 1 when any check failed."""
    rows = readiness_rows(tier=tier, checks=checks, verify_models=verify_models, timeout=timeout)
    table = Table(title=f"SkillEvaluator Doctor (Tier {tier})", box=SIMPLE, show_edge=False)
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Details")
    status_styles = {"pass": "green", "warn": "yellow", "fail": "red"}
    for name, status, detail in rows:
        style = status_styles[status]
        table.add_row(escape_markup(name), f"[{style}]{status}[/{style}]", escape_markup(detail))
    console.print(table)
    return 1 if any(status == "fail" for _, status, _ in rows) else 0
