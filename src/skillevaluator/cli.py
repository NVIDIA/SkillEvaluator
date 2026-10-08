# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line interface for SkillEvaluator."""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import math
import os
import stat
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import click

from skillevaluator import __version__
from skillevaluator.cli_help import GroupedOption, RichGroup
from skillevaluator.constants import (
    SIMILARITY_DEFAULT_MAX_ENTRIES,
    SIMILARITY_DEFAULT_MAX_SCALAR_COMPARISONS,
    SIMILARITY_MAX_ENTRIES,
)
from skillevaluator.logging_config import setup_logging
from skillevaluator.models.result import ValidationResult
from skillevaluator.reporting.console_ui import (
    ValidateView,
    ViewProgressReporter,
    check_ticker_row,
    detail_row,
    engine_feed_rows,
    stage_hint_row,
    summarize_tier1,
    summarize_tier2,
    summarize_tier3,
)
from skillevaluator.reporting.naming import report_basename
from skillevaluator.reporting.plugin_sections import split_display_prefix

# Tier 1 (static validation) is the base install surface and is safe to import
# eagerly. Tier 2 (embeddings/LLM) and Tier 3 (Harbor and its environments)
# pull heavy, extras-only dependencies, so their command implementations are
# imported lazily inside the command callbacks. This keeps `import skillevaluator.cli`
# and the CLI surface available on a base install without those extras.
from skillevaluator.tier1.commands import (
    ReportsNotWrittenError,
    console,
    emit_reports,
    enabled_check_lineup,
    run_lint_scripts,
    run_pii_scan,
    run_quality_check,
    run_rubric_eval,
    run_security_scan,
    run_validation,
)
from skillevaluator.tier3_environments import HARBOR_ENVIRONMENTS, PLUGIN_LOAD_CHOICES
from skillevaluator.tier_group import TierGroup
from skillevaluator.utils.rich_markup import escape_markup, strip_terminal_controls
from skillevaluator.utils.tier2_paths import (
    is_link_or_reparse,
    paths_refer_to_same_location,
    sanitize_tier2_results,
)

if TYPE_CHECKING:
    from skillevaluator.evaluation import EvaluationOptions
    from skillevaluator.tier3.plugin_eval import PluginEvalPackage

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


ENV_MODE_CHOICE = click.Choice(list(HARBOR_ENVIRONMENTS))


class _AliasChoice(click.Choice):
    """Expose current names in help while accepting retired aliases."""

    def __init__(self, choices: list[str], aliases: dict[str, str]) -> None:
        super().__init__(choices)
        self.aliases = aliases

    def convert(self, value, param, ctx):
        if isinstance(value, str):
            value = self.aliases.get(value, value)
        return super().convert(value, param, ctx)


GRADING_MODE_CHOICE = _AliasChoice(
    ["default", "default_plus_custom", "custom_only"],
    {
        "aces_default": "default",
        "aces_plus_custom": "default_plus_custom",
    },
)
CUSTOM_GRADING_MODE_CHOICE = _AliasChoice(
    ["default_plus_custom", "custom_only"],
    {"aces_plus_custom": "default_plus_custom"},
)


def _validate_similarity_threshold(_ctx: click.Context, _param: click.Parameter, value: float) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise click.BadParameter("must be finite and within [0, 1]")
    return value


_PROBE_MCP_ENV_HELP = (
    "Plugin only, with --probe-mcp: a host environment variable the probe may expand into a server's declared "
    "${VAR} headers (repeatable). NAME sends it to every URL server whose headers use it; NAME=HOST only to "
    "servers whose URL host is HOST; NAME@SERVER only to that server. The plugin chooses the URL, so headers "
    "that reference any other variable are not sent."
)


def _validate_probe_mcp_env(_ctx: click.Context, _param: click.Parameter, value: tuple[str, ...]) -> tuple[str, ...]:
    from skillevaluator.tier3.mcp_proof import parse_env_grant

    for item in value:
        try:
            parse_env_grant(item)
        except ValueError as exc:
            raise click.BadParameter(str(exc)) from exc
    return tuple(dict.fromkeys(value))


# Heading + intro for the grouped Tier 3 options in ``validate --help``.
_RUN_GROUP = "Run & Reports"
_RUN_GROUP_DESC = "Applies to the whole run: target typing, policy profile, reports, tier selection."
_TIER1_GROUP = "Tier 1 · Static & Security"
_TIER1_GROUP_DESC = "Static checks; LLM-free by default. Tier 1 gates the exit code and always runs."
_TIER2_GROUP = "Tier 2 · Deduplication"
_TIER2_GROUP_DESC = "Embedding + LLM dedup; on by default, skips gracefully without a provider key."
_TIER3_GROUP = "Tier 3 · Live Agent Evaluation"
_TIER3_GROUP_DESC = "On by default for skills, with automatic dataset preparation; advisory unless made blocking."

# Detailed, sectioned epilog for ``validate --help`` (parity with
# ``skill-evaluator validate -h``). Authored pre-formatted and rendered raw.
_VALIDATE_EPILOG = """
Content types (--type):
  skill      SKILL.md in skills/ or team-skills/
  rules      .mdc files in team-rules/
  workflows  workflow-rules.mdc in a workflow directory
  plugin     Bundle-reference manifest (agent_plugin.yaml/.yml) or
             contained plugin (.claude-plugin/, .codex-plugin/, or
             .cursor-plugin/plugin.json, or an Agent Plugins root plugin.json)

Report formats (-r/--report):
  cli        Rich terminal output (default)
  json       Machine-readable JSON (skillevaluator-output-<timestamp>.json)
  html       Standalone HTML report (skillevaluator-output-<timestamp>.html)
  markdown   Markdown for PR comments (skillevaluator-output-<timestamp>.md)

Tiers:
  Tier 1  Static, security, and quality validation (gates the exit code).
  Tier 2  Embedding similarity + deduplication (on by default; --no-dedup).
  Tier 3  Live agent evaluation (on by default for skills; advisory).
          Reuses an existing dataset or creates one when missing.
          Use --no-tier3 to skip, or --no-autopilot to require an existing dataset.

LLM analysis:
  Configure SKILL_EVAL_LLM_PROVIDER and its provider credential for Tier 2/3.
  Tier 1 LLM security analysis is off by default; enable it with --llm.
  Add --llm-verify for a second pass that suppresses false positives.

Examples:
  skillevaluator validate ./my-skill                        # all three tiers + autopilot
  skillevaluator validate ./my-skill --llm                  # add LLM security scan
  skillevaluator validate ./my-skill -r cli -r json -r html # multiple reports (repeat -r)
  skillevaluator validate ./my-skill -r cli,json,html       # comma-separated too
  skillevaluator validate ./my-skill -o reports/            # custom output dir
  skillevaluator validate ./my-skill --no-dedup             # skip Tier 2 dedup
  skillevaluator validate ./my-skill --external             # strict publish profile
  skillevaluator validate ./my-skill -c                     # continue on failure (record all issues)
  skillevaluator validate ./my-skill -a codex               # choose the Tier 3 agent
  skillevaluator validate ./my-skill --no-autopilot         # require an existing dataset
  skillevaluator validate ./my-skill --tiers 1,2            # omit live evaluation
  skillevaluator validate ./my-skill --tiers 1,3            # explicit tier selection
  skillevaluator validate ./skills-folder                   # whole catalog, serially
  skillevaluator validate ./my-skill -a codex,claude-code \\
      --env-mode docker --harbor-keep-jobs                 # retain Harbor jobs
"""


_TOP_LEVEL_COMMAND_HELP_GROUPS = (
    ("Tier workflows", ("tier1", "tier2", "tier3")),
    ("Core workflows", ("validate", "health-check", "doctor", "models")),
    (
        "Tier 1 · Static and security",
        ("quality-check", "rubric-eval", "security-scan", "pii-scan", "lint-scripts"),
    ),
    (
        "Tier 2 · Deduplication",
        ("similarity-check", "context-optimization-check", "dedup-scan"),
    ),
    (
        "Tier 3 · Live evaluation",
        ("create-eval-dataset", "init-custom-grader", "init-harbor-task", "compare", "view", "harbor-view"),
    ),
)


@click.group(
    cls=RichGroup,
    context_settings=CONTEXT_SETTINGS,
    help_command_groups=_TOP_LEVEL_COMMAND_HELP_GROUPS,
    show_banner=True,
)
@click.version_option(version=__version__, prog_name="skillevaluator")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def cli(verbose: bool) -> None:
    """SKILLEVALUATOR: SkillEvaluator for AI agent skills.

    Three-tier quality gatekeeper for AI agent skills and plugins.

    Documentation: https://docs.nvidia.com/skills/skillevaluator/
    """
    setup_logging(verbose=verbose)


@cli.group(cls=TierGroup)
def tier1() -> None:
    """Run Tier 1 checks: tier1 PATH. LLM checks join automatically when configured.

    With no path, show help. Expert commands remain available below.
    """


@cli.group(cls=TierGroup)
def tier2() -> None:
    """Run Tier 2 deduplication: tier2 PATH [--catalog FILE].

    Compare content within the skill, and against other skills when a saved
    catalog is supplied. With no path, show help.
    """


@cli.group(cls=TierGroup)
def tier3() -> None:
    """Run Tier 3 evaluation: tier3 PATH. Prepare a missing dataset automatically.

    Reuse valid existing tasks; otherwise create one starter case and evaluate
    with and without the skill. With no path, show help.
    """


def _target_argument(func):
    return click.argument("target_path", type=click.Path(exists=True, resolve_path=True, path_type=Path))(func)


def _validate_target_argument(func):
    """Keep the lexical validate root until the default Tier 2 guard runs."""
    return click.argument("target_path", type=click.Path(exists=True, resolve_path=False, path_type=Path))(func)


def _skill_argument(func):
    return click.argument("skill_path", type=click.Path(exists=True, resolve_path=True, path_type=Path))(func)


def _tier2_skill_argument(func):
    """Keep the lexical Tier 2 root so linked roots can be rejected safely."""
    return click.argument("skill_path", type=click.Path(exists=True, resolve_path=False, path_type=Path))(func)


def _reject_linked_tier2_root(path: Path) -> None:
    if is_link_or_reparse(path):
        raise click.UsageError(f"Tier 2 target root is a symlink or reparse point: {path.name or '.'}")


_FILE_REPORT_EXTENSIONS = {
    "json": ".json",
    "html": ".html",
    "markdown": ".md",
    "sarif": ".sarif.json",
}


def _reject_catalog_report_collisions(
    catalog_path: Path | None,
    *,
    report_formats: tuple[str, ...],
    output_dir: Path,
    basename: str,
) -> None:
    if catalog_path is None:
        return
    for report_format in report_formats:
        extension = _FILE_REPORT_EXTENSIONS.get(report_format)
        if extension is None:
            continue
        report_path = output_dir / f"{basename}{extension}"
        if paths_refer_to_same_location(catalog_path, report_path):
            raise click.UsageError(
                f"Catalog path conflicts with the generated {report_format} report: {report_path.name}"
            )


class _MultiValueOption(click.Option):
    """A ``multiple`` option that also accepts comma- and space-separated values.

    Click's native ``multiple`` only supports repeating the flag
    (``-r cli -r json``). This subclass additionally accepts a single flag with
    space-separated values (``-r cli json html``) and comma-separated values
    (``-r cli,json,html``), so all three forms behave identically.

    Tokens after the flag are only consumed while they look like valid choices,
    so a following option or positional argument (e.g. ``-r cli json ./path``)
    cleanly ends the value list. Per-value validation and error messages are
    still produced by the option's ``click.Choice`` type.
    """

    def _looks_like_value(self, raw: str) -> bool:
        choices = getattr(self.type, "choices", None)
        if not choices:
            # Without an explicit choice set we cannot tell a value from a
            # positional, so only accept the single token Click already parsed.
            return False
        parts = [part.strip() for part in raw.split(",")]
        return bool(parts) and all(part in choices for part in parts)

    def add_to_parser(self, parser: click.parser.OptionParser, ctx: click.Context):  # type: ignore[name-defined]
        retval = super().add_to_parser(parser, ctx)

        internal = None
        for opt in self.opts:
            internal = parser._long_opt.get(opt) or parser._short_opt.get(opt)
            if internal is not None:
                break
        if internal is None:
            return retval

        previous_process = internal.process

        def process(value: str, state: click.parser.ParsingState) -> None:  # type: ignore[name-defined]
            tokens = [value]
            # Greedily eat following tokens that still look like report formats.
            while state.rargs and self._looks_like_value(state.rargs[0]):
                tokens.append(state.rargs.pop(0))
            # Append each comma-split value individually so Click's Choice type
            # validates and stores them as a flat sequence.
            for token in tokens:
                for part in token.split(","):
                    part = part.strip()
                    if part:
                        previous_process(part, state)

        internal.process = process  # type: ignore[assignment]
        return retval


def _report_options(func):
    func = click.option(
        "-o",
        "--output-dir",
        type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
        default=Path("reports"),
        show_default=True,
        help="Directory for generated reports.",
    )(func)
    return click.option(
        "-r",
        "--report",
        "report_formats",
        cls=_MultiValueOption,
        multiple=True,
        type=click.Choice(["cli", "json", "html", "markdown", "sarif"]),
        default=("cli",),
        show_default=True,
        help="Report format(s). Accepts comma- or space-separated values "
        "(-r cli,json,html or -r cli json html) and may be repeated. "
        "The compact default view writes html+json unless -r is passed "
        "explicitly, which is honored exactly (including cli).",
    )(func)


_validate_json_report_var: ContextVar[str | None] = ContextVar(
    "_validate_json_report_var",
    default=None,
)


def _effective_report_formats(report_formats: tuple[str, ...], *, quiet: bool) -> tuple[str, ...]:
    """Resolve the report formats ``validate`` will actually emit."""
    if quiet and not _report_formats_explicit():
        return tuple(dict.fromkeys([fmt for fmt in report_formats if fmt != "cli"] + ["html", "json"]))
    return report_formats


def _content_relative_finding_paths(results: list[ValidationResult], validated: Path, content_root: Path) -> None:
    """Make finding paths built from a relative target relative to the content root.

    Validators get the target as typed. For ``validate sample``, some join it
    into their paths ("sample/SKILL.md", relative to the working directory)
    while others report paths relative to the content root ("SKILL.md"), so the
    reports mixed both and SARIF, which reads a relative path against the
    content root, pointed "sample/SKILL.md" nowhere. Each path that starts with
    the typed target is rewritten as ``validate .`` would report it, unless it
    names an existing entry under the content root (a skill "examples" with its
    own "examples/" folder). A bundled skill's ``"[skill] "`` label is kept.
    The ``errors`` / ``warnings`` string that mirrors a moved finding moves
    with it, so reports still match it to the finding and do not list it again.
    """
    prefix = validated.parts
    if validated.is_absolute() or not prefix:
        return  # absolute paths are already unambiguous; "." has no prefix
    for result in results:
        moved: list[tuple[list[str], list[str]]] = []
        for finding in result.findings:
            skill, path = split_display_prefix(finding.file_path or "")
            label = "" if skill is None else f"[{skill}] "
            parts = Path(path).parts
            if parts[: len(prefix)] != prefix:
                continue
            # A path relative to the content root never starts with "..", so only a
            # target without ".." can be ambiguous.
            if ".." not in prefix and (content_root / path).exists(follow_symlinks=False):
                continue
            # Always "/": str(Path(...)) gives "\\" on Windows, and validators there
            # build backslash paths, so reports and SARIF would otherwise differ by OS.
            old_forms = finding.legacy_string_forms()
            finding.file_path = label + ("/".join(parts[len(prefix) :]) or ".")
            moved.append((old_forms, finding.legacy_string_forms()))
        for old_forms, new_forms in moved:
            _move_legacy_string(result, old_forms, new_forms)


def _move_legacy_string(result: ValidationResult, old_forms: list[str], new_forms: list[str]) -> None:
    """Rewrite one ``errors`` / ``warnings`` entry that mirrors a moved finding to its new path."""
    for messages in (result.errors, result.warnings):
        for old, new in zip(old_forms, new_forms, strict=False):
            if old in messages:
                messages[messages.index(old)] = new
                return


def _record_validate_json_report(report_name: str | None) -> None:
    _validate_json_report_var.set(report_name)


def _consume_validate_json_report() -> str | None:
    report_name = _validate_json_report_var.get()
    _validate_json_report_var.set(None)
    return report_name


def _report_formats_explicit() -> bool:
    """True when the user passed ``-r``/``--report`` on the command line.

    Catalog mode re-invokes ``validate`` through a child context that records
    no parameter sources, so the walk climbs to the original invocation.
    """
    from click.core import ParameterSource

    ctx = click.get_current_context(silent=True)
    while ctx is not None:
        source = ctx.get_parameter_source("report_formats")
        if source is not None:
            return source == ParameterSource.COMMANDLINE
        ctx = ctx.parent
    return False


_TIER2_EXTRA_SKIP = "Skipped: install the Tier 2 extra (make install EXTRAS=tier2), or pass --no-dedup."


def _embedding_backend_problem() -> str | None:
    """Why Tier 2 cannot embed here, or ``None`` when it can.

    Embedding needs the ``tier2`` extra (``openai``) and a configured public
    embedding provider. ``find_spec`` does not import the module (so nothing
    leaks into a base install) and may raise if a meta-path blocker is active;
    any failure there counts as the extra missing.
    """
    import importlib.util

    from skillevaluator.provider_config import ProviderConfigurationError, resolve_embedding_provider

    try:
        has_openai = importlib.util.find_spec("openai") is not None
    except (ImportError, ValueError):
        has_openai = False
    if not has_openai:
        return _TIER2_EXTRA_SKIP
    try:
        resolve_embedding_provider()
    except ProviderConfigurationError as exc:
        return f"Tier 2 skipped: embedding setup is required.\n\n{exc}\n\nTo skip Tier 2, pass --no-dedup."
    return None


def _run_dedup_or_skip(target_path: Path) -> list[ValidationResult]:
    """Run Tier 2 dedup when possible, else return a non-failing skipped result.

    Dedup is on by default for ``validate`` but needs the ``tier2`` extra and an
    configured public embedding provider. When either is missing it degrades
    gracefully to a warning so a lightweight ``validate`` keeps working.
    """

    def _skip(message: str) -> list[ValidationResult]:
        result = ValidationResult(
            validator_name="Tier 2 Deduplication",
            validator_description="Embedding-based duplicate detection",
        )
        result.add_warning(message)
        result.metadata["skipped"] = True
        return [result]

    try:
        from skillevaluator.tier2.commands import run_dedup_scan
    except ImportError:
        return _skip(_TIER2_EXTRA_SKIP)
    if problem := _embedding_backend_problem():
        return _skip(problem)
    return run_dedup_scan(target_path)


def _run_plugin_dedup_or_skip(plugin_root: Path) -> list[ValidationResult]:
    """Run the public plugin Tier 2 contract without remote catalog services.

    The embedding-based context checks run only when an embedding backend is
    available; the rest of the contract runs either way.
    """
    from skillevaluator.tier2.commands import run_plugin_dedup_scan

    return run_plugin_dedup_scan(plugin_root, run_context=_embedding_backend_problem() is None)


def _partial_agent_eval_result(
    target_path: Path,
    *,
    engine_result: object,
    failure: str,
    results_dir: Path | None,
    env_mode: str,
    dataset_source: Path | None = None,
    plugin_provenance: dict[str, Any] | None = None,
) -> ValidationResult | None:
    """Normalize an engine run that produced usable results alongside errors.

    A run where one agent crashed but another scored is still a run: discarding
    it as a skip throws away real results. Tier 3 stays advisory -- the errors
    are carried on the result, rendered red in the pipeline view and combined
    report, but never gate the exit code. Returns ``None`` unless the engine
    mapping can be proven to be THIS run's fresh output (in-memory agents data
    plus a ``latest`` results dir matching the engine's ``run_dir``), so a
    stale earlier run is never reported as fresh.
    """
    from skillevaluator.evaluation.tier3_report import agent_eval_result_from_run
    from skillevaluator.tier3.results_location import resolve_latest_results

    if not isinstance(engine_result, dict) or not engine_result.get("agents"):
        return None
    run_dir_value = engine_result.get("run_dir")
    if not run_dir_value or not Path(str(run_dir_value)).is_dir():
        return None
    try:
        latest = resolve_latest_results(target_path, results_dir)
        latest = latest.resolve() if latest.is_symlink() else latest
        if latest.resolve() != Path(str(run_dir_value)).resolve():
            return None
        result = agent_eval_result_from_run(
            target_path,
            results_dir=results_dir,
            env_mode=env_mode,
            engine_result=engine_result,
            **({"dataset_source": dataset_source} if dataset_source is not None else {}),
            **({"plugin_provenance": plugin_provenance} if plugin_provenance is not None else {}),
        )
    except Exception:
        return None
    if result is None:
        return None
    errors = engine_result.get("execution_errors") or engine_result.get("error") or []
    if isinstance(errors, str):
        errors = [errors]
    error_messages = [str(error) for error in errors if str(error).strip()] if isinstance(errors, list) else []
    for message in dict.fromkeys(error_messages or [failure]):
        result.add_error(message)
    return result


def _finalize_evaluated_source(
    results: list[ValidationResult],
    evaluated_source: dict[str, str] | None,
) -> None:
    """Resolve one source identity onto every result before a report is written.

    Provenance is part of every report, not just the card, so it has to be
    finalized ahead of ``emit_reports``. Resolving it afterwards published a JSON
    report that recorded neither the source repository nor its revision while
    BENCHMARK.md recorded both, and let a contradictory identity reach disk as
    JSON and HTML before benchmark generation failed the run.

    The supplied identity is one carrier among the ones the run recorded, not a
    default the results get to override. The fold either raises, when two
    carriers disagree, so a run that cannot say what it evaluated writes nothing
    at all, or yields the union of every carrier, and that union is written back
    so each report reads one complete identity rather than whichever fragment a
    producer happened to record. Leaving a recorded identity in place instead
    let a partial carrier shadow the rest: a result naming only the repository
    published a card calling the revision and the container not recorded, though
    the operator had supplied both.

    The identity rides on the results themselves, for every content type, because
    a PASS can be published without a completed Tier 3 run and so cannot rely on
    the Tier 3 payload as its carrier. A run that recorded no identity anywhere
    has none written, so an absent identity stays absent rather than becoming an
    empty one.
    """
    from skillevaluator.source_identity import (
        EvaluatedSourceConflict,
        merge_evaluated_sources,
        recorded_evaluated_source,
    )

    try:
        identity = merge_evaluated_sources(
            (evaluated_source, recorded_evaluated_source(result.metadata for result in results))
        )
    except EvaluatedSourceConflict as exc:
        # Name the values that disagreed rather than letting a report guess.
        raise click.ClickException(
            f"No report was written because the run records more than one evaluated source ({exc})."
        ) from exc
    if not identity:
        return
    for result in results:
        if isinstance(result.metadata, dict):
            # A fresh dict per result, so mutating one result's identity later
            # cannot rewrite what another result or the caller is holding.
            result.metadata["evaluated_source"] = dict(identity)


def _evaluated_source_from_options(
    repository: str | None,
    revision: str | None,
    container_revision: str | None,
) -> dict[str, str] | None:
    """Build the evaluated-source identity from the orchestration input.

    The identity is supplied rather than inferred, because the tree running the
    evaluator is not the tree being evaluated. A value that is not in its
    canonical shape is rejected here rather than dropped, so a card never says
    ``not recorded`` for a field the operator believed they had supplied.
    """
    from skillevaluator.source_identity import normalized_evaluated_source

    supplied = {
        "repository": repository,
        # One option covers both immutable revision shapes the card accepts,
        # matching its single "Evaluated source revision" line.
        "commit": revision,
        "content_digest": revision,
        "evaluator_container_revision": container_revision,
    }
    identity = normalized_evaluated_source({key: value for key, value in supplied.items() if value}) or {}
    for option, value, accepted, expected in (
        ("--evaluated-source-repository", repository, ("repository",), "a forge name such as owner/repository"),
        (
            "--evaluated-source-revision",
            revision,
            ("commit", "content_digest"),
            "a full Git object id (40 or 64 hex characters) or a digest such as sha256:<64 hex characters>",
        ),
        (
            "--evaluator-container-revision",
            container_revision,
            ("evaluator_container_revision",),
            "an image reference such as ghcr.io/org/image@sha256:<64 hex characters>",
        ),
    ):
        if value and not any(field in identity for field in accepted):
            raise click.BadParameter(f"expected {expected}.", param_hint=option)
    return identity or None


def _run_agent_eval_or_skip(
    target_path: Path,
    *,
    agents: str | None,
    env_mode: str,
    environment_kwarg: tuple[str, ...] = (),
    skip_baseline: bool,
    n_concurrent: int | None,
    max_agents: int | None,
    n_attempts: int | None = None,
    pass_threshold: float | None = None,
    stop_on_pass: bool | None = None,
    model: str | None = None,
    agent_model: tuple[str, ...] = (),
    grading_mode: str | None = None,
    results_dir: Path | None = None,
    include_skills: tuple[Path, ...] = (),
    copy_repo: bool = False,
    timeout_multiplier: float | None = None,
    harbor_keep_jobs: bool = False,
    agent_runtime_preflight: bool | None = None,
    block_on_agent_eval: bool = False,
    validate_source: bool = True,
    evaluated_source: dict[str, str] | None = None,
    progress_reporter=None,
    kind: str = "skill",
    lift_mode: str = "effectiveness",
    repo_root: Path | None = None,
    probe_mcp: bool = False,
    probe_mcp_env: tuple[str, ...] = (),
    allowed_private_hosts: tuple[str, ...] = (),
    plugin_load: str = "wrapper",
    policy: Any = None,
) -> ValidationResult:
    """Run Tier 3 live agent evaluation and fold the result into the combined report.

    Returns an ``AGENT_EVAL`` :class:`ValidationResult` carrying the canonical
    ``metadata["agent_eval"]`` payload on success, or a structured result
    describing why Tier 3 could not run. Tier 3 remains advisory by default,
    and callers can opt into blocking behavior.
    """
    if kind == "plugin":
        return _run_plugin_agent_eval(
            target_path,
            agents=agents,
            env_mode=env_mode,
            environment_kwarg=environment_kwarg,
            skip_baseline=skip_baseline,
            n_concurrent=n_concurrent,
            max_agents=max_agents,
            n_attempts=n_attempts,
            pass_threshold=pass_threshold,
            stop_on_pass=stop_on_pass,
            model=model,
            agent_model=agent_model,
            grading_mode=grading_mode,
            results_dir=results_dir,
            include_skills=include_skills,
            copy_repo=copy_repo,
            timeout_multiplier=timeout_multiplier,
            harbor_keep_jobs=harbor_keep_jobs,
            agent_runtime_preflight=agent_runtime_preflight,
            evaluated_source=evaluated_source,
            progress_reporter=progress_reporter,
            lift_mode=lift_mode,
            repo_root=repo_root,
            probe_mcp=probe_mcp,
            probe_mcp_env=probe_mcp_env,
            allowed_private_hosts=allowed_private_hosts,
            plugin_load=plugin_load,
            policy=policy,
        )

    if validate_source:
        from skillevaluator.evaluation.tier3_report import dataset_required_result
        from skillevaluator.tier3.evals_spec import validate_tier3_source

        source_kind, source_checks = validate_tier3_source(target_path)
        source_errors = [check for check in source_checks if check.status in {"missing", "error"}]
        if source_errors:
            return dataset_required_result(
                target_path / "evals" / "evals.json",
                source_errors,
                blocking=block_on_agent_eval,
                skill_name=target_path.name,
                source_kind=source_kind,
            )

    from skillevaluator.evaluation import EvaluationOptions, EvaluationService
    from skillevaluator.evaluation.tier3_report import (
        advisory_skip_result,
        agent_eval_result_from_run,
    )

    options = EvaluationOptions(
        skill_path=target_path,
        agents=agents,
        env_mode=env_mode,
        environment_kwarg=environment_kwarg,
        skip_baseline=skip_baseline,
        n_concurrent=n_concurrent,
        max_agents=max_agents,
        n_attempts=n_attempts,
        pass_threshold=pass_threshold,
        stop_on_pass=stop_on_pass,
        model=model,
        agent_model=agent_model,
        grading_mode=grading_mode,
        results_dir=results_dir,
        include_skills=include_skills,
        copy_repo=copy_repo,
        timeout_multiplier=timeout_multiplier,
        harbor_keep_jobs=harbor_keep_jobs,
        agent_runtime_preflight=agent_runtime_preflight,
        evaluated_source=evaluated_source,
    )
    try:
        service = EvaluationService()
        if progress_reporter is not None:
            engine_result = service.evaluate(options, progress_reporter=progress_reporter)
        else:
            engine_result = service.evaluate(options)
    except Exception as exc:
        # Preserve a reportable Tier 3 result rather than aborting after the
        # earlier tiers have already completed. The caller's gating metadata
        # decides whether this failure is advisory or blocking.
        return advisory_skip_result(
            f"Tier 3 live evaluation skipped: {exc}",
            skill_name=target_path.name,
        )

    if failure := service.failure_reason(engine_result):
        partial = _partial_agent_eval_result(
            target_path,
            engine_result=engine_result,
            failure=failure,
            results_dir=results_dir,
            env_mode=env_mode,
        )
        if partial is not None:
            return partial
        return advisory_skip_result(
            f"Tier 3 live evaluation did not complete: {failure}",
            skill_name=target_path.name,
        )

    try:
        result = agent_eval_result_from_run(
            target_path,
            results_dir=results_dir,
            env_mode=env_mode,
            engine_result=engine_result if isinstance(engine_result, dict) else None,
        )
    except Exception as exc:
        return advisory_skip_result(
            f"Tier 3 result normalization failed: {exc}",
            skill_name=target_path.name,
        )
    if result is None:
        return advisory_skip_result(
            "Tier 3 live evaluation produced no parseable results.",
            skill_name=target_path.name,
        )
    return result


def _plugin_lift_mode_for_evidence(
    prepared: PluginEvalPackage,
    requested_lift_mode: str,
) -> tuple[str, str | None]:
    """Resolve a plugin lift mode without discarding a valid effectiveness run."""
    if requested_lift_mode not in {"integration", "both"}:
        return requested_lift_mode, None
    evidence_error = prepared.integration_evidence_error()
    if evidence_error and requested_lift_mode == "both":
        return "effectiveness", evidence_error
    return requested_lift_mode, evidence_error


def _plugin_integration_error(lift_mode: str, skip_baseline: bool, reason: str | None) -> str | None:
    """Why a requested plugin Integration comparison cannot run, or ``None``.

    Integration compares against the without-plugin baseline, so it needs that
    arm. ``--lift-mode integration`` also needs composition evidence (*reason*
    says what is missing); ``both`` falls back to effectiveness instead. The
    refusal says the comparison was not run, never a verdict on one.
    """
    if lift_mode not in {"integration", "both"}:
        return None
    if skip_baseline:
        return "Integration requires a baseline; remove --skip-baseline."
    if reason and lift_mode == "integration":
        return f"Integration was not run: {reason}. Fix that, or use --lift-mode effectiveness."
    return None


def _plugin_lift_metadata(
    requested_lift_mode: str,
    effective_lift_mode: str,
    integration_skip_reason: str | None,
) -> dict[str, str]:
    """Record the requested and effective plugin lift modes for reports and provenance.

    Always emitted, not only on a fallback: a report must know that
    ``--lift-mode integration|both`` was asked for to explain an Integration
    comparison that was not measured instead of silently omitting it.
    """
    metadata = {
        "requested_lift_mode": requested_lift_mode,
        "effective_lift_mode": effective_lift_mode,
    }
    if integration_skip_reason is not None:
        metadata["integration_skip_reason"] = integration_skip_reason
    return metadata


# The earlier name, kept for existing imports.
_plugin_lift_fallback_metadata = _plugin_lift_metadata


def _plugin_evaluation_options(
    prepared: PluginEvalPackage,
    *,
    plugin_dir: Path,
    lift_mode: str,
    effective_lift_mode: str,
    integration_skip_reason: str | None,
    plugin_load: str,
    results_dir: Path | None,
    **run_options: Any,
) -> EvaluationOptions:
    """The ``EvaluationOptions`` for a prepared plugin package.

    The plugin-specific fields are set here: the staged package with its member
    skills in one group workspace, the baseline arms the effective lift mode
    needs, the load mode and native snapshot, and the results root of the plugin
    itself rather than of its temporary package. *run_options* are the generic
    Tier 3 options (agents, attempts, models, ...), passed through unchanged.
    """
    from skillevaluator.evaluation import EvaluationOptions
    from skillevaluator.tier3.results_location import resolve_results_root

    return EvaluationOptions(
        skill_path=prepared.package_path,
        skill_workspace_mode="group",
        include_skills=prepared.include_skills,
        workspace_skills_baseline=effective_lift_mode == "integration",
        sum_of_parts_arm=effective_lift_mode == "both",
        eval_target_kind="plugin",
        lift_mode_requested=lift_mode,
        integration_skip_reason=integration_skip_reason,
        plugin_load=plugin_load,
        native_plugin_source=prepared.native_source,
        results_dir=results_dir,
        resolved_results_root=resolve_results_root(plugin_dir, results_dir),
        **run_options,
    )


def _engine_run_dir(engine_result: Any) -> Path | None:
    """The run directory the engine reported, or ``None``."""
    run_dir = engine_result.get("run_dir") if isinstance(engine_result, dict) else None
    return Path(str(run_dir)) if run_dir else None


def _plugin_mcp_proof(
    prepared: PluginEvalPackage,
    *,
    probe_mcp: bool,
    allowed_private_hosts: tuple[str, ...] = (),
    probe_mcp_env: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    """Pre-run MCP proof for author-supplied URL MCP servers, or ``None`` when there are none.

    Without ``--probe-mcp`` every URL server is recorded as ``declared``; with it,
    each gets one bounded host probe (``initialize`` + ``tools/list``) under the
    endpoint policy. Only the host variables named with ``--probe-mcp-env`` (for every
    server, one host, or one server) are expanded into declared headers, and the
    variables that may be sent are printed per host before the probe runs. The
    tool input schemas the probe lists are recorded in the prepared package, so
    the run checks every call's arguments against the server's own schema.
    Advisory only: it never changes the INCOMPLETE rule.
    """
    targets = prepared.mcp_probe_targets
    if not targets:
        return None
    from skillevaluator.tier3.mcp_proof import (
        declared_mcp_proof,
        planned_env_sends,
        probe_mcp_servers,
        write_mcp_input_schemas,
    )

    if not probe_mcp:
        return declared_mcp_proof(targets)
    # Say which opted-in host variables may leave the machine, and to which host, before any probe runs.
    for server, host, names in planned_env_sends(targets, probe_mcp_env):
        console.print(
            f"MCP probe may send {escape_markup(', '.join(names))} to {escape_markup(host or 'an unknown host')} "
            f"(server {escape_markup(server)}, --probe-mcp-env)"
        )
    proof = probe_mcp_servers(targets, allowed_private_hosts=allowed_private_hosts, expand_env=probe_mcp_env)
    write_mcp_input_schemas(getattr(prepared, "package_path", None), proof)
    return proof


def _incomplete_plugin_provenance(
    prepared: PluginEvalPackage,
    engine_result: Any,
    mcp_proof: dict[str, Any] | None,
    failure: str,
    metadata: dict[str, str],
) -> dict[str, Any] | None:
    """Plugin provenance for a run that did not complete, written to the run dir before any error.

    The with-plugin arm's load census, hook census, canary, and coverage are
    still evidence even when another arm (or some trials) failed, so they are
    kept and the run is marked INCOMPLETE instead of being dropped.
    """
    from skillevaluator.tier3.plugin_eval import write_plugin_provenance

    try:
        provenance = _plugin_provenance_with_runtime_evidence(prepared, engine_result, mcp_proof)
    except Exception:
        return None
    provenance.update(metadata)
    provenance["execution_incomplete"] = f"Tier 3 plugin evaluation did not complete: {failure}"[:2000]
    provenance["partial"] = True
    run_dir = _engine_run_dir(engine_result)
    if run_dir is not None and run_dir.is_dir():
        # Best effort: the in-memory provenance still reaches the result and reports.
        with contextlib.suppress(Exception):
            write_plugin_provenance(run_dir, provenance)
    return provenance


def _completed_plugin_provenance(
    prepared: PluginEvalPackage,
    engine_result: Any,
    mcp_proof: dict[str, Any] | None,
    metadata: dict[str, str],
) -> dict[str, Any]:
    """Plugin provenance for a completed run, also written to the run dir as its sidecar.

    The runner rendered ``report.html`` before the sidecar existed, so the
    caller refreshes the report once the result is known.
    """
    from skillevaluator.tier3.plugin_eval import write_plugin_provenance

    provenance = _plugin_provenance_with_runtime_evidence(prepared, engine_result, mcp_proof)
    provenance.update(metadata)
    run_dir = _engine_run_dir(engine_result)
    if run_dir is not None:
        write_plugin_provenance(run_dir, provenance)
    return provenance


def _plugin_provenance_with_runtime_evidence(
    prepared: PluginEvalPackage, engine_result: Any, mcp_proof: dict[str, Any] | None
) -> dict[str, Any]:
    """Plugin provenance plus runtime coverage (``exercised``) and the MCP proof."""
    from skillevaluator.tier3.plugin_native import finalize_native_provenance
    from skillevaluator.tier3.plugin_runtime import apply_runtime_evidence

    provenance = prepared.provenance()
    if mcp_proof is not None:
        provenance["mcp_proof"] = mcp_proof
    # Native load census first: runtime evidence reads provenance["plugin_load"].
    finalize_native_provenance(provenance, engine_result)
    return apply_runtime_evidence(provenance, engine_result if isinstance(engine_result, dict) else None)


def _incomplete_plugin_agent_eval_result(
    plugin_dir: Path,
    *,
    prepared: PluginEvalPackage,
    engine_result: Any,
    failure: str,
    mcp_proof: dict[str, Any] | None,
    metadata: dict[str, str],
    results_dir: Path | None,
    env_mode: str,
) -> ValidationResult:
    """An INCOMPLETE ``AGENT_EVAL`` result for a plugin run that did not complete.

    Writes ``plugin_provenance.json`` first, then builds the result from this
    run's own output so the with-plugin canary, load census, hook census, and
    coverage stay in every report. Falls back to an advisory skip only when the
    run left no usable results.
    """
    from skillevaluator.evaluation.tier3_report import (
        advisory_skip_result,
        incomplete_reason,
        refresh_plugin_run_report,
    )

    message = f"Tier 3 plugin evaluation did not complete: {failure}"
    provenance = _incomplete_plugin_provenance(prepared, engine_result, mcp_proof, failure, metadata)
    result = _partial_agent_eval_result(
        plugin_dir,
        engine_result=engine_result,
        failure=message,
        results_dir=results_dir,
        env_mode=env_mode,
        dataset_source=prepared.package_path,
        plugin_provenance=provenance,
    )
    if result is None:
        return advisory_skip_result(message, skill_name=plugin_dir.name)
    result.passed = False
    result.metadata["execution_status"] = "skipped"
    result.metadata["skip_reason"] = incomplete_reason(provenance or {"execution_incomplete": message})
    result.metadata.update(metadata)
    run_dir = _engine_run_dir(engine_result)
    if run_dir is not None:
        # The runner rendered report.html before the sidecar existed.
        refresh_plugin_run_report(plugin_dir, run_dir, result=result)
    return result


def _incomplete_plugin_skip_result(plugin_dir: Path, prepared: Any) -> ValidationResult:
    """An INCOMPLETE ``AGENT_EVAL`` result for a plugin with nothing locally evaluable.

    A plugin whose only refs are external, missing, or unresolved, or whose
    only component is a provider-only MCP server, evaluated nothing. That is
    not an advisory skip: the result fails, says INCOMPLETE, and carries the
    plugin provenance, so every report lists what could not be evaluated.
    """
    from skillevaluator.evaluation.tier3_report import advisory_skip_result, incomplete_reason

    provenance = prepared.provenance()
    reason = incomplete_reason(provenance)
    message = f"Tier 3 plugin evaluation is INCOMPLETE: nothing was evaluated. {prepared.skip_reason or ''}".strip()
    result = advisory_skip_result(message, skill_name=plugin_dir.name)
    payload = result.metadata.get("agent_eval")
    if isinstance(payload, dict):
        payload["plugin_provenance"] = provenance
        payload["provenance"] = {"source": "plugin", "reason": "incomplete", "advisory": False, "message": message}
    result.passed = False
    result.metadata["execution_status"] = "skipped"
    result.metadata["skip_reason"] = f"{reason} (nothing was evaluated)"
    return result


def _run_plugin_agent_eval(
    plugin_target: Path,
    *,
    agents: str | None,
    env_mode: str,
    environment_kwarg: tuple[str, ...] = (),
    skip_baseline: bool,
    n_concurrent: int | None,
    max_agents: int | None,
    n_attempts: int | None = None,
    pass_threshold: float | None = None,
    stop_on_pass: bool | None = None,
    model: str | None = None,
    agent_model: tuple[str, ...] = (),
    grading_mode: str | None = None,
    results_dir: Path | None = None,
    include_skills: tuple[Path, ...] = (),
    copy_repo: bool = False,
    timeout_multiplier: float | None = None,
    harbor_keep_jobs: bool = False,
    agent_runtime_preflight: bool | None = None,
    evaluated_source: dict[str, str] | None = None,
    progress_reporter=None,
    lift_mode: str = "effectiveness",
    repo_root: Path | None = None,
    probe_mcp: bool = False,
    probe_mcp_env: tuple[str, ...] = (),
    allowed_private_hosts: tuple[str, ...] = (),
    plugin_load: str = "wrapper",
    policy: Any = None,
) -> ValidationResult:
    """Stage and evaluate a public plugin without fetching remote components."""
    import tempfile

    from skillevaluator.cli_core import resolve_plugin_path
    from skillevaluator.evaluation import EvaluationService
    from skillevaluator.evaluation.tier3_report import (
        advisory_skip_result,
        agent_eval_result_from_run,
        refresh_plugin_run_report,
    )
    from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package

    plugin_dir = resolve_plugin_path(plugin_target)

    def _skipped(message: str) -> ValidationResult:
        return advisory_skip_result(message, skill_name=plugin_dir.name)

    lift_metadata: dict[str, str] = {}
    try:
        with tempfile.TemporaryDirectory(prefix="skillevaluator-plugin-eval-") as temp_dir:
            prepared = prepare_plugin_eval_package(
                plugin_dir,
                stage_root=Path(temp_dir),
                include_skills=include_skills,
                repo_root=repo_root,
                plugin_load=plugin_load,
                agents=agents,
                env_mode=env_mode,
                policy=policy,
            )
            if prepared.skipped or prepared.package_path is None:
                if getattr(prepared, "incomplete_skip", False):
                    return _incomplete_plugin_skip_result(plugin_dir, prepared)
                return _skipped(
                    f"Tier 3 plugin evaluation skipped: {prepared.skip_reason or 'nothing locally evaluable'}"
                )
            effective_lift_mode, integration_skip_reason = _plugin_lift_mode_for_evidence(prepared, lift_mode)
            if integration_error := _plugin_integration_error(lift_mode, skip_baseline, integration_skip_reason):
                return _skipped(f"Tier 3 plugin {integration_error}")
            lift_metadata = _plugin_lift_metadata(lift_mode, effective_lift_mode, integration_skip_reason)
            mcp_proof = _plugin_mcp_proof(
                prepared,
                probe_mcp=probe_mcp,
                allowed_private_hosts=allowed_private_hosts,
                probe_mcp_env=probe_mcp_env,
            )
            options = _plugin_evaluation_options(
                prepared,
                plugin_dir=plugin_dir,
                lift_mode=lift_mode,
                effective_lift_mode=effective_lift_mode,
                integration_skip_reason=integration_skip_reason,
                plugin_load=plugin_load,
                results_dir=results_dir,
                agents=agents,
                env_mode=env_mode,
                environment_kwarg=environment_kwarg,
                skip_baseline=skip_baseline,
                n_concurrent=n_concurrent,
                max_agents=max_agents,
                n_attempts=n_attempts,
                pass_threshold=pass_threshold,
                stop_on_pass=stop_on_pass,
                model=model,
                agent_model=agent_model,
                grading_mode=grading_mode,
                copy_repo=copy_repo,
                timeout_multiplier=timeout_multiplier,
                harbor_keep_jobs=harbor_keep_jobs,
                agent_runtime_preflight=agent_runtime_preflight,
                evaluated_source=evaluated_source,
            )
            service = EvaluationService()
            if progress_reporter is not None:
                engine_result = service.evaluate(options, progress_reporter=progress_reporter)
            else:
                engine_result = service.evaluate(options)
            if failure := service.failure_reason(engine_result):
                return _incomplete_plugin_agent_eval_result(
                    plugin_dir,
                    prepared=prepared,
                    engine_result=engine_result,
                    failure=failure,
                    mcp_proof=mcp_proof,
                    metadata=lift_metadata,
                    results_dir=results_dir,
                    env_mode=env_mode,
                )
            provenance = _completed_plugin_provenance(prepared, engine_result, mcp_proof, lift_metadata)
            result = agent_eval_result_from_run(
                plugin_dir,
                results_dir=results_dir,
                dataset_source=prepared.package_path,
                env_mode=env_mode,
                engine_result=engine_result if isinstance(engine_result, dict) else None,
                plugin_provenance=provenance,
            )
            run_dir = _engine_run_dir(engine_result)
            if result is not None and run_dir is not None:
                # The runner rendered report.html before the sidecar existed.
                refresh_plugin_run_report(plugin_dir, run_dir, result=result)
    except Exception as exc:
        return _skipped(f"Tier 3 plugin evaluation skipped: {exc}")

    if result is None:
        return _skipped("Tier 3 plugin evaluation produced no parseable results.")
    result.metadata.update(lift_metadata)
    return result


# Per-tier section headings printed by ``validate`` as each tier runs. They give
# the CLI/CI stream the same progressive, labeled structure SkillEvaluator emitted, so
# Tier 1 (and Tier 2) are visibly reported as they execute instead of only
# surfacing in the single combined report rendered at the very end.
_TIER_BANNERS = {
    "tier1": "Tier 1: Security and Static Validation",
    "tier2": "Tier 2: Deduplication",
    "tier3": "Tier 3: Live Agent Evaluation",
}


def _ensure_autopilot_dataset(skill_path: Path, *, quiet: bool = False, progress: str = "auto") -> str | None:
    """Ensure an evaluation source exists, generating one when missing.

    Mirrors the standalone ``evaluate --autopilot`` behavior: reuse an existing
    source unchanged; otherwise generate one case with the configured provider,
    falling back to a deterministic case when no provider key is available.
    Returns a short note describing what happened (for the pipeline view), or
    raises ``click.ClickException`` when generation cannot produce a source.
    """
    from skillevaluator.evaluation import EvaluationService
    from skillevaluator.evaluation.results import DatasetGenerationError
    from skillevaluator.inference.diagnostics import llm_failure_diagnostic, safe_llm_labels
    from skillevaluator.provider_config import ProviderConfigurationError, resolve_llm_provider
    from skillevaluator.tier3.dataset_progress import dataset_generation_progress
    from skillevaluator.tier3.harbor.adapter import find_evals_file

    def echo(message: str) -> None:
        if not quiet:
            click.echo(message, err=True)

    def eval_source_exists() -> bool:
        return find_evals_file(skill_path) is not None or (skill_path / "evals" / "harbor").exists()

    service = EvaluationService()
    try:
        if eval_source_exists():
            echo("Autopilot: reusing the existing evaluation source unchanged.")
            return "existing evaluation source — autopilot generation not needed"
        try:
            provider = resolve_llm_provider()
        except ProviderConfigurationError:
            no_llm = True
            echo("Autopilot: no public provider key is configured; generating one deterministic case.")
        else:
            no_llm = False
            provider_name, model_name = safe_llm_labels(provider.provider, provider.model)
            echo(f"Autopilot: generating one case with {provider_name} / {model_name}.")

        dataset_note = "auto-generated by autopilot — review evals/"
        with dataset_generation_progress(
            enabled=not quiet and progress != "off", stream=click.get_text_stream("stderr")
        ):
            try:
                service.create_autopilot_dataset(skill_path, use_llm=not no_llm)
            except FileExistsError:
                echo("Autopilot: an evaluation source appeared concurrently; reusing it unchanged.")
            except Exception as exc:
                if no_llm:
                    raise
                cause = exc.__cause__ if isinstance(exc, DatasetGenerationError) else None
                diagnostic = llm_failure_diagnostic(cause if isinstance(cause, Exception) else exc)
                echo(f"Warning: Autopilot LLM generation failed.\n  Reason: {diagnostic}")
                echo("  Falling back to one deterministic starter case; review evals/ before using it as a benchmark.")
                dataset_note = f"deterministic starter — LLM generation failed: {diagnostic} Review evals/."
                if not eval_source_exists():
                    service.create_autopilot_dataset(skill_path, use_llm=False)
            if not eval_source_exists():
                raise click.ClickException("Autopilot dataset generation did not produce an evaluation source.")
        return dataset_note
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


def _print_catalog_divider(index: int, total: int, name: str) -> None:
    """A full-width rule announcing the next per-skill job in a catalog run."""
    from rich.text import Text

    from skillevaluator.reporting.console_ui import FAINT, GREEN, WIDTH, make_view_console

    console_ = make_view_console()
    width = max(60, min(WIDTH, console_.width))
    label = Text.assemble(
        ("━━ ", FAINT),
        (f"skill {index}/{total}", f"bold {GREEN}"),
        (" · ", FAINT),
        (strip_terminal_controls(name), "bold"),
        (" ", ""),
    )
    fill = max(0, width - label.cell_len)
    console_.print()
    console_.print(label + Text("━" * fill, style=FAINT))
    console_.print()


def _print_catalog_summary(total: int, failures: list[tuple[str, str]], reports_root: Path) -> None:
    """The catalog scoreboard: per-skill verdict counts and where reports live."""
    from rich import box
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    from skillevaluator.reporting.console_ui import (
        GREEN,
        MUTED,
        RED,
        TEXT,
        WIDTH,
        make_view_console,
    )

    console_ = make_view_console()
    width = max(60, min(WIDTH, console_.width))
    passed = total - len(failures)
    if failures:
        headline = Text.assemble(
            (f" ✗ {len(failures)} FAILED ", f"bold #1C0605 on {RED}"),
            ("  ", ""),
            (f"{passed}/{total} skills passed", f"bold {TEXT}"),
        )
        lines: list = [headline, Text()]
        for name, reason in failures[:10]:
            # Strip before padding and truncating, so controls neither print nor count toward the widths.
            name_text = f"{strip_terminal_controls(name):<28}"
            lines.append(
                Text.assemble(("  ✗ ", f"bold {RED}"), (name_text, TEXT), (strip_terminal_controls(reason)[:56], MUTED))
            )
        if len(failures) > 10:
            lines.append(Text(f"  … {len(failures) - 10} more — see per-skill reports", style=MUTED))
        body = Group(*lines)
        border = RED
    else:
        body = Text.assemble(
            (" ✓ PASS ", f"bold #101403 on {GREEN}"),
            ("  ", ""),
            (f"all {total} skills passed", f"bold {TEXT}"),
        )
        border = GREEN
    console_.print()
    console_.print(
        Panel(
            body,
            box=box.ROUNDED,
            border_style=border,
            width=width,
            padding=(0, 2),
            title="[bold]Catalog Result[/bold]",
            title_align="left",
        )
    )
    reports = f"{strip_terminal_controls(str(reports_root))}/<skill>/"
    console_.print(Text.assemble(("      reports     ", MUTED), (reports, MUTED)))


CATALOG_SUMMARY_FILENAME = "catalog-summary.json"


def _catalog_child_argv_from_ctx(ctx: click.Context, skill_dir: Path, output_dir: Path) -> list[str]:
    """Rebuild ``validate`` argv from the active Click context (pytest-safe)."""
    params = ctx.params
    argv: list[str] = ["validate", str(skill_dir)]

    if params.get("verbose"):
        argv.append("--verbose")
    if params.get("full"):
        argv.append("--full")
    if params.get("tiers"):
        argv.extend(["--tiers", str(params["tiers"])])
    if params.get("checks"):
        argv.extend(["--checks", str(params["checks"])])
    if params.get("previous_version"):
        argv.extend(["--previous-version", str(params["previous_version"])])
    if params.get("fail_fast"):
        argv.append("--fail-fast")
    if params.get("continue_on_failure"):
        argv.append("-c")
    if params.get("llm"):
        argv.append("--llm")
    else:
        argv.append("--no-llm")
    if params.get("llm_verify"):
        argv.append("--llm-verify")
    if not params.get("dedup", True):
        argv.append("--no-dedup")
    block_on_dedup = params.get("block_on_dedup")
    if block_on_dedup is True:
        argv.append("--block-on-dedup")
    elif block_on_dedup is False:
        argv.append("--no-block-on-dedup")
    min_score = params.get("min_score", 70)
    if min_score != 70:
        argv.extend(["--min-score", str(min_score)])
    if params.get("external"):
        argv.append("--external")
    if params.get("policy_path"):
        argv.extend(["--policy", str(params["policy_path"])])
    if params.get("profile"):
        argv.extend(["--profile", str(params["profile"])])
    if params.get("repo_root"):
        argv.extend(["--repo-root", str(params["repo_root"])])
    if params.get("resolve_endpoints"):
        argv.append("--resolve-endpoints")
    if params.get("agent_eval") is True:
        argv.append("--tier3")
    elif params.get("agent_eval") is False:
        argv.append("--no-tier3")
    block_on_agent_eval = params.get("block_on_agent_eval")
    if block_on_agent_eval is True:
        argv.append("--block-on-agent-eval")
    elif block_on_agent_eval is False:
        argv.append("--no-block-on-agent-eval")
    if params.get("autopilot") is True:
        argv.append("--autopilot")
    elif params.get("autopilot") is False:
        argv.append("--no-autopilot")
    agents = params.get("agents")
    if agents:
        argv.extend(["--agents", str(agents)])
    env_mode = params.get("env_mode", "docker")
    if env_mode != "docker":
        argv.extend(["--env-mode", str(env_mode)])
    for environment_kwarg in params.get("environment_kwarg") or ():
        argv.extend(["--environment-kwarg", str(environment_kwarg)])
    lift_mode = params.get("lift_mode", "effectiveness")
    if lift_mode != "effectiveness":
        argv.extend(["--lift-mode", str(lift_mode)])
    if params.get("probe_mcp"):
        argv.append("--probe-mcp")
    for name in params.get("probe_mcp_env") or ():
        argv.extend(["--probe-mcp-env", str(name)])
    plugin_load = params.get("plugin_load", "wrapper")
    if plugin_load != "wrapper":
        argv.extend(["--plugin-load", str(plugin_load)])
    if params.get("skip_baseline"):
        argv.append("--skip-baseline")
    if params.get("n_concurrent") is not None:
        argv.extend(["--n-concurrent", str(params["n_concurrent"])])
    if params.get("max_agents") is not None:
        argv.extend(["--max-agents", str(params["max_agents"])])
    if params.get("n_attempts") is not None:
        argv.extend(["--n-attempts", str(params["n_attempts"])])
    if params.get("pass_threshold") is not None:
        argv.extend(["--pass-threshold", str(params["pass_threshold"])])
    stop_on_pass = params.get("stop_on_pass")
    if stop_on_pass is True:
        argv.append("--stop-on-pass")
    elif stop_on_pass is False:
        argv.append("--no-stop-on-pass")
    if params.get("model"):
        argv.extend(["--model", str(params["model"])])
    for override in params.get("agent_model") or ():
        argv.extend(["--agent-model", str(override)])
    if params.get("grading_mode"):
        argv.extend(["--grading-mode", str(params["grading_mode"])])
    if params.get("results_dir"):
        argv.extend(["--results-dir", str(params["results_dir"])])
    for skill in params.get("include_skills") or ():
        argv.extend(["--include-skills", str(skill)])
    if params.get("copy_repo"):
        argv.append("--copy-repo")
    if params.get("timeout_multiplier") is not None:
        argv.extend(["--timeout-multiplier", str(params["timeout_multiplier"])])
    if params.get("harbor_keep_jobs"):
        argv.append("--harbor-keep-jobs")
    agent_runtime_preflight = params.get("agent_runtime_preflight")
    if agent_runtime_preflight is True:
        argv.append("--agent-runtime-preflight")
    elif agent_runtime_preflight is False:
        argv.append("--no-agent-runtime-preflight")
    # Every child renders its own BENCHMARK.md, so the identity has to reach
    # each one: dropping it here would publish a catalog of cards that all say
    # the evaluated source was never recorded.
    if params.get("evaluated_source_repository"):
        argv.extend(["--evaluated-source-repository", str(params["evaluated_source_repository"])])
    if params.get("evaluated_source_revision"):
        argv.extend(["--evaluated-source-revision", str(params["evaluated_source_revision"])])
    if params.get("evaluator_container_revision"):
        argv.extend(["--evaluator-container-revision", str(params["evaluator_container_revision"])])
    if _report_formats_explicit():
        for fmt in params.get("report_formats") or ():
            argv.extend(["-r", fmt])
    argv.extend(["-o", str(output_dir)])
    return argv


def _run_catalog_skill_worker(job: dict[str, Any]) -> tuple[str, bool, str, str | None]:
    """Run one catalog skill validation in a child process."""
    import os

    from click.testing import CliRunner

    workdir = job.get("cwd")
    if workdir:
        os.chdir(workdir)

    skill_name = str(job["skill_name"])
    result = CliRunner().invoke(cli, job["argv"])
    json_report_name = _consume_validate_json_report()
    if result.exit_code == 0:
        return skill_name, True, "", json_report_name
    exc = result.exception
    if isinstance(exc, SystemExit):
        return skill_name, False, "validation failed", json_report_name
    if exc is not None:
        if isinstance(exc, click.ClickException):
            return skill_name, False, str(getattr(exc, "message", exc)), json_report_name
        return skill_name, False, f"unexpected error: {exc}", json_report_name
    return skill_name, False, "validation failed", json_report_name


def _catalog_skill_entry(
    skill_name: str,
    skill_report_dir: Path,
    *,
    passed: bool,
    reason: str,
    json_report_name: str | None = None,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "name": skill_name,
        "passed": passed,
        "report_dir": skill_name,
    }
    if not passed:
        entry["reason"] = reason

    if json_report_name:
        json_report = skill_report_dir / json_report_name
    else:
        return entry

    if not json_report.is_file():
        return entry

    entry["json_report"] = json_report.name
    try:
        payload = json.loads(json_report.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return entry
    if not isinstance(payload, dict):
        return entry

    for key in ("overall_passed", "overall_status", "incomplete_scans", "severity_counts"):
        if key in payload:
            entry[key] = payload[key]
    return entry


def _write_catalog_summary(output_dir: Path, skills: list[dict[str, object]]) -> Path:
    """Write a machine-readable fleet rollup for catalog validation."""
    from skillevaluator.reporting.base import _write_report_atomically

    total = len(skills)
    passed = sum(1 for skill in skills if skill.get("passed"))
    failed = total - passed
    severity_totals = {
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
    }
    for skill in skills:
        counts = skill.get("severity_counts")
        if not isinstance(counts, dict):
            continue
        for key in severity_totals:
            value = counts.get(key)
            if isinstance(value, int):
                severity_totals[key] += value

    summary: dict[str, object] = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "overall_passed": failed == 0,
        "reports_root": output_dir.name,
        "summary_path": CATALOG_SUMMARY_FILENAME,
        "severity_totals": severity_totals,
        "skills": skills,
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }
    output_path = output_dir / CATALOG_SUMMARY_FILENAME
    payload = json.dumps(summary, indent=2, default=str, allow_nan=False).encode("utf-8")
    _write_report_atomically(output_path, payload)
    return output_path


def _validate_catalog(
    ctx: click.Context,
    *,
    skill_dirs: list[Path],
    output_dir: Path,
    workers: int = 1,
) -> None:
    """Run the full validate pipeline once per skill in the catalog.

    Each skill is an independent job with its own pipeline view, reports
    (under ``<output_dir>/<skill>/``), and verdict; the catalog exits nonzero
    when any skill failed. With ``workers`` above 1, skills validate in
    parallel child processes and the per-skill pipeline view is skipped.
    """
    from skillevaluator.validators.version import PREVIOUS_VERSION_ENV

    # The environment variable is the flag's default, so it gets the same guard.
    if ctx.params.get("previous_version") or os.environ.get(PREVIOUS_VERSION_ENV):
        raise click.ClickException(
            f"--previous-version (or {PREVIOUS_VERSION_ENV}) applies to one skill and cannot be reused for a "
            "catalog; validate each skill separately with its own previous version"
        )
    if workers > 1:
        _validate_catalog_parallel(ctx, skill_dirs=skill_dirs, output_dir=output_dir, workers=workers)
        return

    failures: list[tuple[str, str]] = []
    skill_reports: dict[str, str | None] = {}
    for index, skill_dir in enumerate(skill_dirs, start=1):
        _print_catalog_divider(index, len(skill_dirs), skill_dir.name)
        skill_output = output_dir / skill_dir.name
        overrides = {
            **ctx.params,
            "target_path": skill_dir,
            "content_type": "skill",
            "output_dir": skill_output,
        }
        reason = ""
        try:
            ctx.invoke(validate, **overrides)
        except click.ClickException as exc:
            reason = str(getattr(exc, "message", exc))
            failures.append((skill_dir.name, reason))
        except Exception as exc:  # unexpected: keep the catalog running, report it on the scoreboard
            reason = f"unexpected error: {exc}"
            failures.append((skill_dir.name, reason))
        skill_reports[skill_dir.name] = _consume_validate_json_report()
    failure_map = dict(failures)
    skill_entries = [
        _catalog_skill_entry(
            skill_dir.name,
            output_dir / skill_dir.name,
            passed=skill_dir.name not in failure_map,
            reason=failure_map.get(skill_dir.name, ""),
            json_report_name=skill_reports.get(skill_dir.name),
        )
        for skill_dir in skill_dirs
    ]
    _write_catalog_summary(output_dir, skill_entries)
    _print_catalog_summary(len(skill_dirs), failures, output_dir)
    if failures:
        raise click.ClickException(
            f"{len(failures)}/{len(skill_dirs)} skills failed validation: "
            + ", ".join(strip_terminal_controls(name) for name, _reason in failures)
        )


def _validate_catalog_parallel(
    ctx: click.Context,
    *,
    skill_dirs: list[Path],
    output_dir: Path,
    workers: int,
) -> None:
    """Validate catalog skills concurrently in isolated child processes."""
    from skillevaluator.reporting.console_ui import make_view_console

    if not skill_dirs:
        _write_catalog_summary(output_dir, [])
        _print_catalog_summary(0, [], output_dir)
        return

    workdir = str(Path.cwd())
    make_view_console().print(
        f"[dim]Validating {len(skill_dirs)} skills with {workers} worker"
        f"{'s' if workers != 1 else ''} (parallel catalog mode; per-skill pipeline view disabled)[/dim]"
    )

    jobs = [
        {
            "skill_name": skill_dir.name,
            "argv": _catalog_child_argv_from_ctx(ctx, skill_dir, output_dir / skill_dir.name),
            "cwd": workdir,
            "output_dir": str(output_dir / skill_dir.name),
        }
        for skill_dir in skill_dirs
    ]
    failures: list[tuple[str, str]] = []
    worker_reports: dict[str, str | None] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_catalog_skill_worker, job) for job in jobs]
        for future in as_completed(futures):
            skill_name, passed, reason, json_report_name = future.result()
            worker_reports[skill_name] = json_report_name
            if not passed:
                failures.append((skill_name, reason))

    failures.sort(key=lambda item: item[0])
    failure_map = dict(failures)
    skill_entries = [
        _catalog_skill_entry(
            skill_dir.name,
            output_dir / skill_dir.name,
            passed=skill_dir.name not in failure_map,
            reason=failure_map.get(skill_dir.name, ""),
            json_report_name=worker_reports.get(skill_dir.name),
        )
        for skill_dir in skill_dirs
    ]
    _write_catalog_summary(output_dir, skill_entries)
    _print_catalog_summary(len(skill_dirs), failures, output_dir)
    if failures:
        raise click.ClickException(
            f"{len(failures)}/{len(skill_dirs)} skills failed validation: "
            + ", ".join(strip_terminal_controls(name) for name, _reason in failures)
        )


def _rerun_hint(target_path: Path, agent_eval: bool) -> str:
    """Reconstruct the user's actual command for the FAIL panel's rerun line.

    A bare ``skillevaluator validate <path>`` would drop the flags that shaped
    the failing run (--min-score, --profile, --checks, ...), so following it
    could silently "pass" the failure away. Falls back to the bare form when
    the process was not launched as the CLI (tests, API embedding).
    """
    import shlex
    import sys

    executable = Path(sys.argv[0] or "").name
    if executable.startswith("skillevaluator") and len(sys.argv) > 1:
        return shlex.join([executable, *sys.argv[1:]])
    return f"skillevaluator validate {target_path}" + (" --tier3" if agent_eval else "")


def _finish_pipeline_view(
    view: ValidateView,
    *,
    tier_gate_results: list[ValidationResult],
    tier3_result: ValidationResult | None,
    gate_failed: bool,
    output_dir: Path,
    basename: str,
    report_formats: tuple[str, ...],
    target_path: Path,
    agent_eval: bool,
) -> None:
    """Render the quiet-mode verdict panel and report footer."""
    from skillevaluator.reporting.console_ui import Verdict, _is_skipped, first_fix

    ext = {"html": ".html", "json": ".json", "markdown": ".md", "sarif": ".sarif.json"}
    links: list[tuple[str, str]] = [
        ("report" if fmt == "html" else fmt, str(output_dir / f"{basename}{ext[fmt]}"))
        for fmt in report_formats
        if fmt in ext
    ]
    payload = ((tier3_result.metadata or {}).get("agent_eval") or {}) if tier3_result is not None else {}
    summary = payload.get("summary") or payload
    # A Tier 3 FAIL (complete or partial run) or an INCOMPLETE partial run that stayed advisory.
    tier3_gate = _tier3_gate_label(tier3_result) or (
        "Tier 3 INCOMPLETE"
        if tier3_result is not None and not tier3_result.passed and _gate_result_incomplete(tier3_result)
        else None
    )

    if not gate_failed:
        ran = sum(1 for block in view.blocks if block.status not in ("pending", "skip"))
        advisory_failed = any(block.status == "fail" and block.number in {2, 3} for block in view.blocks)
        if tier3_gate:
            # Tier 3 failed its gate or is INCOMPLETE, but it is advisory here: never call that "all tiers passed".
            headline = f"gating tiers passed · {tier3_gate} is advisory (--block-on-agent-eval gates it)"
        elif advisory_failed:
            headline = "gating tiers passed · advisory tier reported findings (see report)"
        else:
            headline = f"all {ran} tier{'s' if ran != 1 else ''} passed"
            lift = summary.get("overall_lift")
            if agent_eval and isinstance(lift, (int, float)):
                headline += f"  ·  skill lift {lift:+.2f}"
        view.finish(Verdict(passed=True, headline=headline), links)
        return

    blocked = [result for result in tier_gate_results if not result.passed]
    incomplete = [result for result in blocked if _gate_result_incomplete(result)]
    # A skipped Tier 3 run that gates is a failure (it did not run); other skipped results never block.
    failed = [
        result
        for result in blocked
        if not _gate_result_incomplete(result) and (result.validator_name == "AGENT_EVAL" or not _is_skipped(result))
    ]
    # Missing evidence also counts when the same result failed on a blocking finding of its own.
    missing = [result for result in blocked if result in incomplete or result.is_incomplete]
    rerun = _rerun_hint(target_path, agent_eval)
    if failed:
        # A real failure outranks missing evidence; the headline still counts what did not complete.
        headline = _gate_headline(failed, "failed")
        if missing:
            headline += f" · {len(missing)} did not complete"
        verdict = Verdict(passed=False, headline=headline, fix=first_fix(tier_gate_results), rerun=rerun)
    elif incomplete:
        verdict = Verdict(
            passed=False, incomplete=True, headline=_gate_headline(incomplete, "did not complete"), rerun=rerun
        )
    else:
        verdict = Verdict(passed=False, headline="validation failed", fix=first_fix(tier_gate_results), rerun=rerun)
    view.finish(verdict, links)


def _tier3_gate_label(result: ValidationResult | None) -> str | None:
    """Return why a Tier 3 run, complete or partial, failed its gate (``Tier 3 verdict FAIL``), or ``None``."""
    labels = (result.metadata or {}).get("tier3_gate_failures") if result is not None else None
    if not isinstance(labels, list) or not labels:
        return None
    return " · ".join(str(label) for label in labels)


def _gate_result_incomplete(result: ValidationResult) -> bool:
    """Whether a blocking result is only missing evidence (INCOMPLETE) rather than failed.

    The rule is the one the terminal summary and BENCHMARK.md use. A scanner that
    did not complete is INCOMPLETE, unless the same result also recorded a
    blocking finding of its own: a real failure outranks missing evidence. A
    partial Tier 3 plugin run is INCOMPLETE unless it failed its gate (a FAIL
    verdict or a confirmed Skill Lift regression). A Tier 3 run that did not run
    at all fails the gate when ``--block-on-agent-eval`` makes Tier 3 gate.
    """
    from skillevaluator.reporting.benchmark import has_blocking_finding
    from skillevaluator.reporting.console_ui import _is_skipped

    if result.validator_name != "AGENT_EVAL":
        return result.is_incomplete and not has_blocking_finding(result)
    metadata = result.metadata or {}
    payload = metadata.get("agent_eval") if isinstance(metadata.get("agent_eval"), dict) else {}
    if _tier3_gate_label(result) or str(payload.get("verdict") or "").lower() == "fail":
        return False
    if _tier3_not_run_message(result) is not None:
        return False
    return _is_skipped(result)


def _tier3_not_run_message(result: ValidationResult) -> str | None:
    """Return why Tier 3 did not run at all (an advisory skip result), or ``None``."""
    payload = (result.metadata or {}).get("agent_eval")
    provenance = payload.get("provenance") if isinstance(payload, dict) else None
    if not isinstance(provenance, dict) or not (provenance.get("advisory") and provenance.get("reason") == "skipped"):
        return None
    return str(provenance.get("message") or "Tier 3 live evaluation did not run").strip()


def _gate_headline(results: list[ValidationResult], outcome: str) -> str:
    """Name the first blocking result, its tier, and how many more share the outcome."""
    first = results[0]
    extra = f" (+{len(results) - 1} more)" if len(results) > 1 else ""
    tier_no = ((first.metadata or {}).get("gating") or {}).get("tier")
    tier_label = f" in Tier {tier_no}" if tier_no is not None else ""
    if first.validator_name == "AGENT_EVAL":
        if label := _tier3_gate_label(first):
            return f"{label}{extra}"
        if outcome == "did not complete":
            reason = str((first.metadata or {}).get("skip_reason") or "").strip()
            reason = reason.removeprefix("INCOMPLETE:").strip()
            return f"Tier 3{extra}: {reason}" if reason else f"Tier 3{extra} did not complete"
        if (message := _tier3_not_run_message(first)) is not None:
            # A gated Tier 3 run that never ran: say why (the message already names Tier 3).
            return f"{message}{extra}" if message.startswith("Tier 3") else f"Tier 3 did not run{extra}: {message}"
        return f"Tier 3 live evaluation{extra} {outcome}"
    return f"{first.validator_name or 'validation'}{extra} {outcome}{tier_label}"


def _print_tier_banner(title: str) -> None:
    """Print a labeled per-tier section banner (parity with SkillEvaluator)."""
    click.echo(f"\n{title}")
    click.echo("-" * 50)


def _print_run_banner(target_path: Path, content_type: str, profile: str | None) -> None:
    """Print the pre-run header (target + detected type + active profile).

    Restores parity with SkillEvaluator's ``_print_validation_banner``: before any
    tier runs, surface what is being validated, the resolved content type, and
    the active validation profile so CI logs and terminal sessions identify the
    run up front instead of opening straight on the Tier 1 section.
    """
    console.print(f"\n[bold]SkillEvaluator {content_type.title()} Validation[/bold]")
    console.print(f"Target: {escape_markup(str(target_path))}")
    console.print(f"Type: {content_type}")
    if profile:
        profile_color = "cyan"
        console.print(f"Profile: [{profile_color}]{escape_markup(str(profile))}[/{profile_color}]")


def _declared_target_is_link(target_path: Path) -> bool:
    """Whether the validation target is a symlink or reparse point; a hard-linked or special one is refused."""
    from skillevaluator.utils.secure_fs import stat_is_link_or_reparse

    try:
        metadata = target_path.lstat()
    except OSError as exc:
        raise click.ClickException(f"Cannot inspect validation target safely: {exc}") from exc
    is_link = stat_is_link_or_reparse(metadata)
    if stat.S_ISREG(metadata.st_mode) and getattr(metadata, "st_nlink", 1) != 1:
        raise click.UsageError(f"Validation target is a hard-linked file: {target_path.name or '.'}")
    if not is_link and not (stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode)):
        raise click.UsageError(f"Validation target is not a regular file or directory: {target_path.name or '.'}")
    return is_link


def _has_regular_skill_manifest(directory: Path) -> bool:
    """Whether *directory* holds a regular, single-link skill manifest, checked without following links."""
    from skillevaluator.constants import SKILL_MANIFEST_VARIANTS
    from skillevaluator.utils.secure_fs import stat_is_link_or_reparse

    for manifest_name in SKILL_MANIFEST_VARIANTS:
        try:
            metadata = (directory / manifest_name).lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise click.ClickException(f"Cannot inspect validation target safely: {exc}") from exc
        if (
            not stat_is_link_or_reparse(metadata)
            and stat.S_ISREG(metadata.st_mode)
            and getattr(metadata, "st_nlink", 1) == 1
        ):
            return True
    return False


class _ValidateTarget(NamedTuple):
    """The content ``validate`` runs on."""

    content_type: str
    root: Path
    #: The direct child skills of a catalog (a directory of skills without a root manifest); otherwise empty.
    catalog_skill_dirs: list[Path]


def _resolve_validate_target(
    target_path: Path,
    *,
    content_type: str,
    detected_type: str,
    declared_is_link: bool,
    tier1_only: bool,
) -> _ValidateTarget:
    """Resolve the content type and root ``validate`` runs on, and the skills of a catalog.

    A linked target root is followed only for a Tier 1-only run of a
    directory, never when the target names a selected manifest; an
    auto-detected type is then detected again on the resolved root. A
    directory of skills without a regular root ``SKILL.md`` is a catalog, and
    its direct child skills are returned so each one is validated as its own
    job.
    """
    from skillevaluator.cli_core import detect_content_type, resolve_content_path
    from skillevaluator.constants import (
        CONTENT_TYPE_SKILL,
        CONTENT_TYPE_UNKNOWN,
        PLUGIN_CONTAINED_MANIFEST_FILE,
        PLUGIN_MANIFEST_FILES,
        RULES_FILE_EXTENSION,
        SKILL_MANIFEST_VARIANTS,
    )

    resolved_type = detected_type
    resolved_target = resolve_content_path(target_path, resolved_type)
    if declared_is_link:
        names_selected_manifest = (
            target_path.name in SKILL_MANIFEST_VARIANTS
            or target_path.name in PLUGIN_MANIFEST_FILES
            # Any plugin.json: vendor-directory manifests and the Agent Plugins root manifest.
            or target_path.name == PLUGIN_CONTAINED_MANIFEST_FILE
            or target_path.suffix == RULES_FILE_EXTENSION
        )
        if names_selected_manifest or not tier1_only or not target_path.is_dir():
            raise click.UsageError(
                f"Validation target root is a symlink or reparse point (including a junction): "
                f"{target_path.name or '.'}"
            )
        try:
            resolved_target = target_path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise click.ClickException(f"Cannot resolve linked Tier 1 validation root safely: {exc}") from exc
        if content_type == "auto":
            resolved_type = detect_content_type(resolved_target)
            resolved_target = resolve_content_path(resolved_target, resolved_type)

    catalog_skill_dirs: list[Path] = []
    if (
        resolved_type in (CONTENT_TYPE_SKILL, CONTENT_TYPE_UNKNOWN)
        and resolved_target.is_dir()
        and not _has_regular_skill_manifest(resolved_target)
    ):
        from skillevaluator.utils.helpers import find_skills_in_directory

        try:
            discovered = find_skills_in_directory(resolved_target)
        except ValueError as exc:
            raise click.ClickException(f"Cannot discover validation target safely: {exc}") from exc
        if resolved_target not in discovered:
            catalog_skill_dirs = sorted(skill_dir for skill_dir in discovered if skill_dir.parent == resolved_target)
    return _ValidateTarget(resolved_type, resolved_target, catalog_skill_dirs)


@cli.command(epilog=_VALIDATE_EPILOG)
@_validate_target_argument
@click.option(
    "--type",
    "content_type",
    default="auto",
    show_default=True,
    type=click.Choice(["skill", "rules", "workflows", "plugin", "auto"]),
    cls=GroupedOption,
    help_group=_RUN_GROUP,
    help="Force the content type instead of auto-detecting it from the target path.",
)
@click.option(
    "--tiers",
    default=None,
    cls=GroupedOption,
    help_group=_RUN_GROUP,
    help="Explicit tier selection, e.g. --tiers 1,3. Tier 1 always runs.",
)
@click.option(
    "--full",
    is_flag=True,
    cls=GroupedOption,
    help_group=_RUN_GROUP,
    help="Compatibility alias for all three tiers with autopilot (already the default for skills). "
    "Explicit tier selection and disable flags take precedence.",
)
@click.option(
    "--verbose",
    "verbose",
    is_flag=True,
    cls=GroupedOption,
    help_group=_RUN_GROUP,
    help="Print the full per-check detail stream instead of the compact pipeline view.",
)
@click.option(
    "--checks",
    "--tier1-checks",
    "checks",
    cls=GroupedOption,
    help_group=_TIER1_GROUP,
    help="Comma-separated subset of Tier 1 checks to run (default: all applicable). "
    "Choices: schema, version, security, pii, license, code-integrity, unicode, quality, lint; "
    "opt-in (not run by default): dependency, and claude-validate (plugins: parity with "
    "'claude plugin validate' when the claude CLI is installed). "
    "quality/lint/version are skill-only and skipped for rules/workflows.",
)
@click.option(
    "--previous-version",
    default=None,
    metavar="VERSION",
    cls=GroupedOption,
    help_group=_TIER1_GROUP,
    help="Previous released version for strictly increasing SemVer validation. "
    "Can also be supplied via SKILLEVALUATOR_PREVIOUS_VERSION.",
)
@click.option(
    "--fail-fast",
    is_flag=True,
    cls=GroupedOption,
    help_group=_TIER1_GROUP,
    help="Stop on the first failing check instead of collecting all issues.",
)
@click.option(
    "-c",
    "--continue-on-failure",
    "continue_on_failure",
    is_flag=True,
    cls=GroupedOption,
    help_group=_TIER1_GROUP,
    help="Run the full pipeline without stopping early; record all issues in the reports. "
    "Overrides --fail-fast, and for folder validation keeps scanning every skill past a "
    "CRITICAL finding.",
)
@click.option(
    "--llm/--no-llm",
    "--tier1-llm/--no-tier1-llm",
    "llm",
    default=False,
    show_default=True,
    cls=GroupedOption,
    help_group=_TIER1_GROUP,
    help="Enable LLM-backed security analysis (requires a configured public provider).",
)
@click.option(
    "--llm-verify",
    is_flag=True,
    cls=GroupedOption,
    help_group=_TIER1_GROUP,
    help="Run a second LLM pass to suppress false-positive findings.",
)
@click.option(
    "--min-score",
    type=int,
    default=70,
    show_default=True,
    cls=GroupedOption,
    help_group=_TIER1_GROUP,
    help="Minimum quality score (0-100) required to pass when the 'quality' check runs.",
)
@click.option(
    "--profile",
    default=None,
    cls=GroupedOption,
    help_group=_RUN_GROUP,
    help="Validation profile: external or a custom name. Default: $SKILLEVALUATOR_PROFILE env var, then external.",
)
@click.option(
    "--external",
    is_flag=True,
    cls=GroupedOption,
    help_group=_RUN_GROUP,
    help="Shortcut for --profile external (validate for public publication).",
)
@click.option(
    "--policy",
    "policy_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    cls=GroupedOption,
    help_group=_RUN_GROUP,
    help="Custom policy YAML overlaid on top of --profile.",
)
@click.option(
    "--workers",
    type=click.IntRange(1),
    default=1,
    show_default=True,
    cls=GroupedOption,
    help_group=_RUN_GROUP,
    help="Concurrent catalog skill jobs when validating a folder of skills. "
    "Values above 1 run skills in parallel processes and disable the per-skill pipeline view.",
)
@click.option(
    "--repo-root",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    cls=GroupedOption,
    help_group=_RUN_GROUP,
    help="Plugin only: repository root (git top-level) used to resolve same-repository skill/rule "
    "references in Tier 1 and Tier 3. Default: the git top-level containing the plugin. The "
    "missing-dependency gate also requires that root to have a git 'origin' remote; otherwise "
    "references stay unresolved (advisory).",
)
@click.option(
    "--resolve-endpoints",
    is_flag=True,
    cls=GroupedOption,
    help_group=_TIER1_GROUP,
    help="Plugin only, opt-in network check: resolve MCP and HTTP hook URL hosts and send one "
    "credential-free HEAD (no redirects followed) to flag names or redirects that reach private, "
    "link-local, or cloud-metadata addresses. Also enabled by 'endpoints.resolve: true' in the policy. "
    "Default: off (Tier 1 stays network-free).",
)
@click.option(
    "--dedup/--no-dedup",
    "--tier2/--no-tier2",
    "dedup",
    default=True,
    show_default=True,
    cls=GroupedOption,
    help_group=_TIER2_GROUP,
    help="Run Tier 2 intra-skill semantic-overlap checks. On by default; skipped "
    "gracefully without public embedding access. Use --no-tier2 (or --no-dedup) to disable.",
)
@click.option(
    "--block-on-dedup/--no-block-on-dedup",
    default=None,
    cls=GroupedOption,
    help_group=_TIER2_GROUP,
    help="Make Tier 2 findings gate the exit code. Default: blocking.",
)
@click.option(
    "--tier3/--no-tier3",
    "--agent-eval/--no-agent-eval",
    "agent_eval",
    default=None,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Run Tier 3 live agent evaluation. On by default for skills; --no-tier3 also disables dataset generation.",
)
@click.option(
    "--block-on-agent-eval/--no-block-on-agent-eval",
    default=None,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Make Tier 3 gate the exit code: a FAIL verdict, a confirmed Skill Lift regression (FAIL band, whole "
    "interval below zero), or a run that is skipped or INCOMPLETE then fails validate. A NEUTRAL verdict never "
    "does. Default: advisory (reported, exit code unchanged).",
)
@click.option(
    "--autopilot/--no-autopilot",
    default=None,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Reuse the evaluation dataset or generate one when missing. On by default for skills; "
    "--no-autopilot requires an existing dataset. Explicit --no-tier3 or --tiers can exclude Tier 3.",
)
@click.option(
    "-a",
    "--agents",
    default=None,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help=(
        "Comma-separated Harbor agents. Default follows the provider: "
        "NVIDIA Build=opencode, OpenAI=codex, Anthropic=claude-code."
    ),
)
@click.option(
    "--env-mode",
    default="docker",
    show_default=True,
    type=ENV_MODE_CHOICE,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Harbor environment backend.",
)
@click.option(
    "--lift-mode",
    type=click.Choice(["effectiveness", "integration", "both"]),
    default="effectiveness",
    show_default=True,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Plugin only: compare against no plugin, sum-of-parts, or both baselines.",
)
@click.option(
    "--probe-mcp",
    is_flag=True,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Plugin only: before Tier 3, probe author-supplied URL MCP servers from the host "
    "(initialize + tools/list, bounded) under the endpoint policy. Sends only literal declared headers "
    "unless --probe-mcp-env names a variable. Advisory.",
)
@click.option(
    "--probe-mcp-env",
    "probe_mcp_env",
    multiple=True,
    metavar="NAME[=HOST|@SERVER]",
    callback=_validate_probe_mcp_env,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help=_PROBE_MCP_ENV_HELP,
)
@click.option(
    "--plugin-load",
    type=click.Choice(PLUGIN_LOAD_CHOICES),
    default="wrapper",
    show_default=True,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Plugin only: how the with-plugin arm loads the plugin. 'wrapper' stages a generated wrapper skill; "
    "'native' stages it the way each harness loads plugins (fails for unsupported agents or local mode); "
    "'auto' uses native where supported and the wrapper otherwise.",
)
@click.option(
    "--environment-kwarg",
    "--ek",
    multiple=True,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Harbor environment constructor kwarg, KEY=VALUE. Repeat for multiple values; never pass secrets.",
)
@click.option(
    "--skip-baseline",
    is_flag=True,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Skip the without-skill baseline in live eval (no lift analysis, faster).",
)
@click.option(
    "--n-concurrent",
    type=int,
    default=None,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Concurrent eval cases per agent.",
)
@click.option(
    "--max-agents",
    type=int,
    default=None,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Maximum agents to run in parallel.",
)
@click.option(
    "--n-attempts",
    type=int,
    default=None,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Attempts per eval case (pass@k).",
)
@click.option(
    "--pass-threshold",
    type=float,
    default=None,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Score threshold (0.0-1.0) for a case to count as passed.",
)
@click.option(
    "--stop-on-pass/--no-stop-on-pass",
    default=None,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Stop a case's remaining attempts once one passes.",
)
@click.option(
    "--model",
    default=None,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Global agent model override.",
)
@click.option(
    "--agent-model",
    multiple=True,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Per-agent model override, AGENT=MODEL (repeatable).",
)
@click.option(
    "--grading-mode",
    type=GRADING_MODE_CHOICE,
    default=None,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Reward/grading mode for live eval.",
)
@click.option(
    "--results-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Directory for Harbor live-eval results.",
)
@click.option(
    "--include-skills",
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Additional skill(s) to mount into the eval environment (repeatable).",
)
@click.option(
    "--copy-repo",
    is_flag=True,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Copy the surrounding repo into the eval environment.",
)
@click.option(
    "--timeout-multiplier",
    type=float,
    default=None,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Scale Harbor step timeouts.",
)
@click.option(
    "--harbor-keep-jobs",
    is_flag=True,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Retain Harbor job dirs/artifacts after the run for inspection.",
)
@click.option(
    "--agent-runtime-preflight/--no-agent-runtime-preflight",
    default=None,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Run an extra agent-only execution of the first staged task before measured Tier 3 [default: disabled].",
)
@click.option(
    "--evaluated-source-repository",
    default=None,
    cls=GroupedOption,
    help_group=_RUN_GROUP,
    help="Repository (owner/name) of the source tree being evaluated, recorded on BENCHMARK.md.",
)
@click.option(
    "--evaluated-source-revision",
    default=None,
    cls=GroupedOption,
    help_group=_RUN_GROUP,
    help="Immutable revision of the evaluated source: a full Git object id, or a sha256/sha384/sha512 digest.",
)
@click.option(
    "--evaluator-container-revision",
    default=None,
    cls=GroupedOption,
    help_group=_RUN_GROUP,
    help="Digest-pinned evaluator image reference recorded beside the evaluated source.",
)
@_report_options
def validate(
    target_path: Path,
    content_type: str,
    tiers: str | None,
    full: bool,
    verbose: bool,
    checks: str | None,
    previous_version: str | None,
    fail_fast: bool,
    continue_on_failure: bool,
    llm: bool,
    llm_verify: bool,
    min_score: int,
    profile: str | None,
    external: bool,
    policy_path: Path | None,
    repo_root: Path | None,
    resolve_endpoints: bool,
    dedup: bool,
    block_on_dedup: bool | None,
    agent_eval: bool | None,
    block_on_agent_eval: bool | None,
    autopilot: bool | None,
    agents: str | None,
    env_mode: str,
    lift_mode: str,
    probe_mcp: bool,
    probe_mcp_env: tuple[str, ...],
    plugin_load: str,
    environment_kwarg: tuple[str, ...],
    skip_baseline: bool,
    n_concurrent: int | None,
    max_agents: int | None,
    n_attempts: int | None,
    pass_threshold: float | None,
    stop_on_pass: bool | None,
    model: str | None,
    agent_model: tuple[str, ...],
    grading_mode: str | None,
    results_dir: Path | None,
    include_skills: tuple[Path, ...],
    copy_repo: bool,
    timeout_multiplier: float | None,
    harbor_keep_jobs: bool,
    agent_runtime_preflight: bool | None,
    workers: int,
    evaluated_source_repository: str | None,
    evaluated_source_revision: str | None,
    evaluator_container_revision: str | None,
    report_formats: tuple[str, ...],
    output_dir: Path,
) -> None:
    """Validate a skill through all three tiers, with automatic dataset preparation.

    Runs Tier 1 static, security, and quality checks (which gate the exit code),
    blocking Tier 2 deduplication, and advisory Tier 3 live evaluation.
    For skills, reuse the existing dataset or generate one when missing.
    Use --tiers 1,2 to omit live evaluation, or --block-on-agent-eval to make
    Tier 3 gate the exit code: a Tier 3 FAIL verdict, a confirmed Skill Lift
    regression, or a skipped or INCOMPLETE Tier 3 run then fails validate; NEUTRAL
    does not. Without the flag, Tier 3 is reported (BENCHMARK.md says it was
    advisory) but never changes the exit code.
    Reports follow --report and --output-dir.

    Rules, workflows, and plugins retain Tier 1/2 by default; their Tier 3
    evaluation requires an explicit --tier3, --autopilot, or --full request.

    A plugin (a bundle-reference ``agent_plugin.yaml``/``.yml`` manifest, or a
    contained ``.claude-plugin/``, ``.codex-plugin/``, or ``.cursor-plugin/``
    ``plugin.json`` or Agent Plugins root ``plugin.json``) is auto-detected and
    validated against its public contract; quality/lint/version checks run on
    each skill bundled under the plugin's ``skills/`` directory.
    """
    _record_validate_json_report(None)
    evaluated_source = _evaluated_source_from_options(
        evaluated_source_repository,
        evaluated_source_revision,
        evaluator_container_revision,
    )

    from skillevaluator.cli_core import detect_content_type
    from skillevaluator.constants import (
        CONTENT_TYPE_PLUGIN,
        CONTENT_TYPE_RULES,
        CONTENT_TYPE_SKILL,
        CONTENT_TYPE_WORKFLOWS,
    )
    from skillevaluator.reporting import CLIReporter
    from skillevaluator.reporting.naming import REPORT_PREFIX
    from skillevaluator.utils.helpers import make_timestamped_basename, resolve_git_remote_url, resolve_git_root
    from skillevaluator.validators.policy import apply_policy, resolve_policy

    declared_is_link = _declared_target_is_link(target_path)

    if external and profile and profile != "external":
        raise click.ClickException(f"--external conflicts with --profile {profile}; pass one or the other.")
    profile_name = "external" if external else profile
    try:
        policy = resolve_policy(profile=profile_name, policy_path=policy_path)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    block_on_dedup_effective = True if block_on_dedup is None else block_on_dedup
    block_on_agent_eval_effective = False if block_on_agent_eval is None else block_on_agent_eval

    detected_type = content_type if content_type != "auto" else detect_content_type(target_path)
    # Only skills own an evals/ task source. Plugins, rules, and workflows can
    # still run Tier 3, but must bypass the skill-directory source preflight.
    preflight_tier3_source = detected_type == CONTENT_TYPE_SKILL

    # Skill validation is the complete workflow. Keep the old flags as aliases,
    # while explicit disable flags and the authoritative --tiers selector still
    # let callers narrow the work. Resolve per child for catalog validation.
    if agent_eval is None:
        agent_eval = preflight_tier3_source or full or autopilot is True
    if autopilot is None:
        autopilot = preflight_tier3_source or full
    if tiers:
        selected = {part.strip() for part in tiers.split(",") if part.strip()}
        unknown = sorted(selected - {"1", "2", "3"})
        if unknown:
            raise click.ClickException(f"--tiers accepts 1, 2, and 3; got: {', '.join(unknown)}")
        if "1" not in selected:
            raise click.ClickException("Tier 1 always runs and gates the exit code; include it (e.g. --tiers 1,3).")
        dedup = dedup and "2" in selected
        # --tiers is authoritative in both directions: an explicit selection
        # turns Tier 3 off even when --full/--autopilot/--tier3 turned it on.
        agent_eval = "3" in selected
    autopilot = autopilot and agent_eval

    run_tier2 = dedup and detected_type in (CONTENT_TYPE_SKILL, CONTENT_TYPE_PLUGIN)
    run_tier3 = agent_eval and detected_type in (CONTENT_TYPE_SKILL, CONTENT_TYPE_PLUGIN)
    resolved_type, resolved_target, catalog_skill_dirs = _resolve_validate_target(
        target_path,
        content_type=content_type,
        detected_type=detected_type,
        declared_is_link=declared_is_link,
        # Auto-detection through the link could still turn Tier 2 or 3 on, so it counts as requesting them.
        tier1_only=not (run_tier2 or run_tier3 or (content_type == "auto" and (dedup or agent_eval))),
    )
    # A directory of skills (no root SKILL.md) is a catalog: run the pipeline
    # once per skill, each as its own job with its own reports.
    if catalog_skill_dirs:
        _validate_catalog(
            click.get_current_context(),
            skill_dirs=catalog_skill_dirs,
            output_dir=output_dir,
            workers=workers,
        )
        return

    # Quiet (default) drives the compact pipeline view; --verbose keeps the
    # historical full-detail stream, as does DEBUG logging via the group -v.
    quiet = not verbose and not logging.getLogger().isEnabledFor(logging.DEBUG)
    planned_tiers = [(1, "Static & Security", "static & security")]
    tier2_index = tier3_index = None
    if run_tier2:
        tier2_index = len(planned_tiers)
        planned_tiers.append((2, "Deduplication", "deduplication"))
    if run_tier3:
        tier3_index = len(planned_tiers)
        planned_tiers.append((3, "Live Agent Eval", "live agent eval"))
    # The lexical target may be "." or a manifest file; name reports after the resolved content root.
    target_name = resolved_target.resolve().name
    view = ValidateView(
        skill=f"{resolved_type}: {target_name}",
        tiers=planned_tiers,
        command="validate",
        enabled=quiet,
    )
    if quiet:
        # Tool/scan narration is debug detail; the view narrates the run.
        logging.disable(logging.INFO)
        ctx = click.get_current_context()
        ctx.call_on_close(lambda: logging.disable(logging.NOTSET))
        ctx.call_on_close(view.stop)
    else:
        _print_run_banner(target_path, resolved_type, getattr(policy, "profile", None))
        _print_tier_banner(_TIER_BANNERS["tier1"])

    view.start()
    view.tier_start(0)
    check_lineup = enabled_check_lineup(checks)
    checks_done: list[str] = []

    def _on_check(name: str) -> None:
        view.tier_progress(0, [check_ticker_row(check_lineup, checks_done, name)])
        checks_done.append(name)

    results = run_validation(
        resolved_target,
        checks=checks,
        use_llm=llm,
        llm_verify=llm_verify,
        min_score=min_score,
        previous_version=previous_version,
        policy=policy,
        content_type=resolved_type,
        fail_fast=fail_fast,
        continue_on_failure=continue_on_failure,
        on_check=_on_check if quiet else None,
        repo_root=repo_root,
        resolve_endpoints=resolve_endpoints,
    )
    if resolved_type == CONTENT_TYPE_PLUGIN and resolved_target != target_path:
        from skillevaluator.validators.plugin_schema import PluginSchemaValidator

        PluginSchemaValidator.note_requested_manifest(results, target_path)
    # The raw pass/fail signal drives --fail-fast identically in both modes;
    # the DISPLAYED tier summary must reflect policy-finalized severities or
    # the tier blocks can contradict the verdict panel (apply_policy is
    # idempotent, so emit_reports re-applying it later is a no-op).
    tier1_raw_failed = any(not r.passed for r in results)
    if quiet:
        apply_policy(results, policy)
    tier1_ok, tier1_rows = summarize_tier1(results, lineup=check_lineup)
    view.tier_done(0, failed=not tier1_ok, rows=tier1_rows)
    tier1_gate_results = list(results)
    tier2_gate_results: list[ValidationResult] = []

    if run_tier2 and not (fail_fast and not continue_on_failure and tier1_raw_failed):
        if not quiet:
            _print_tier_banner(_TIER_BANNERS["tier2"])
        view.tier_start(tier2_index)
        view.tier_progress(tier2_index, [stage_hint_row("stages", "chunk · embed · cluster · llm-judge")])
        tier2_results = (
            _run_plugin_dedup_or_skip(resolved_target)
            if resolved_type == CONTENT_TYPE_PLUGIN
            else _run_dedup_or_skip(resolved_target)
        )
        results.extend(tier2_results)
        tier2_gate_results.extend(tier2_results)
        if quiet:
            apply_policy(tier2_results, policy)
        tier2_ran, tier2_ok, tier2_rows, tier2_skip = summarize_tier2(tier2_results)
        if tier2_ran:
            view.tier_done(tier2_index, failed=not tier2_ok, rows=tier2_rows)
        else:
            view.tier_skip(tier2_index, tier2_skip)
    elif run_tier2:
        view.tier_skip(tier2_index, "skipped after Tier 1 failure (fail-fast)")

    # Preserve the pre-Tier 3 results for progressive output. Final gate
    # membership is resolved from the explicit tri-state options below.
    tier_gate_results = [*tier1_gate_results, *tier2_gate_results]

    # Flush Tier 1 + Tier 2 results to the terminal BEFORE the long-running
    # Tier 3 agent evaluation so they stay visible in CI logs even when Tier 3
    # is slow, errors, or is interrupted before the combined report is emitted.
    # Severities are finalized first so this interim view matches the combined
    # report rendered at the end (apply_policy is idempotent, so emit_reports
    # re-applying it is a no-op).
    if not quiet and run_tier3 and "cli" in report_formats:
        apply_policy(tier_gate_results, policy)
        CLIReporter(console=console).print_summary(tier_gate_results)

    # Tier 3 runs BEFORE report emission so its results are folded into the
    # single combined HTML/JSON/BENCHMARK.md report (parity with SkillEvaluator), and
    # runs regardless of Tier 1/Tier 2 outcome. It degrades to a non-blocking
    # advisory note when it cannot run.
    tier3_result: ValidationResult | None = None
    if run_tier3:
        if not quiet:
            _print_tier_banner(_TIER_BANNERS["tier3"])
        view.tier_start(tier3_index)
        env_note = {
            "docker": "isolated containers per trial",
            "local": "experimental host sandbox — trusted skills and workspaces only",
        }.get(env_mode, "")
        model_display = ", ".join(agent_model) if agent_model else (model or "agent defaults")
        tier3_config_rows = [
            detail_row("agent", agents or "provider-native default"),
            detail_row("env", env_mode, env_note),
            detail_row("model", model_display),
        ]

        # Autopilot: reuse the standalone evaluate command's dataset flow.
        # Tier 3 is advisory, so a dataset-generation failure must not abort
        # validate after Tier 1/2 already ran -- Tier 3 skips with the reason.
        autopilot_error: str | None = None
        if autopilot:
            view.tier_progress(
                tier3_index, [*tier3_config_rows, stage_hint_row("status", "preparing evaluation dataset…")]
            )
            try:
                dataset_note = _ensure_autopilot_dataset(resolved_target, quiet=quiet)
            except (Exception, SystemExit) as exc:
                autopilot_error = f"autopilot dataset generation failed: {getattr(exc, 'message', exc)}"
                if not quiet:
                    click.echo(f"Warning: {autopilot_error}", err=True)
            else:
                if dataset_note:
                    tier3_config_rows.append(detail_row("dataset", dataset_note))

        view.tier_progress(
            tier3_index,
            [*tier3_config_rows, stage_hint_row("status", "running with-skill and baseline trials…")],
        )

        def _on_engine_tail(lines: list[str]) -> None:
            view.tier_progress(tier3_index, [*tier3_config_rows, *engine_feed_rows(lines)])

        reporter = ViewProgressReporter(_on_engine_tail) if quiet else None
        tier3_result = _run_agent_eval_or_skip(
            resolved_target,
            agents=agents,
            env_mode=env_mode,
            environment_kwarg=environment_kwarg,
            skip_baseline=skip_baseline,
            n_concurrent=n_concurrent,
            max_agents=max_agents,
            n_attempts=n_attempts,
            pass_threshold=pass_threshold,
            stop_on_pass=stop_on_pass,
            model=model,
            agent_model=agent_model,
            grading_mode=grading_mode,
            results_dir=results_dir,
            include_skills=include_skills,
            copy_repo=copy_repo,
            timeout_multiplier=timeout_multiplier,
            harbor_keep_jobs=harbor_keep_jobs,
            agent_runtime_preflight=agent_runtime_preflight,
            block_on_agent_eval=block_on_agent_eval_effective,
            validate_source=preflight_tier3_source,
            evaluated_source=evaluated_source,
            progress_reporter=reporter,
            kind=resolved_type,
            lift_mode=lift_mode,
            repo_root=repo_root,
            probe_mcp=probe_mcp,
            probe_mcp_env=probe_mcp_env,
            allowed_private_hosts=tuple(policy.mcp_allowed_private_hosts),
            plugin_load=plugin_load,
            policy=policy,
        )
        results.append(tier3_result)
        tier3_ran, tier3_ok, tier3_rows, tier3_skip = summarize_tier3(tier3_result)
        if autopilot_error and not tier3_ran:
            tier3_skip = f"{autopilot_error}; {tier3_skip}"
            # Reports read the skip reason from metadata, so the generation
            # failure must land there too, not only in the view's skip row.
            tier3_result.metadata["skip_reason"] = tier3_skip
        if not tier3_ran and (tier3_gate_label := _tier3_gate_label(tier3_result)):
            # A partial run that failed its gate (FAIL verdict or confirmed regression) is FAIL, not skipped,
            # as in the footer and the reports; the note says what was not evaluated.
            view.tier_done(
                tier3_index,
                failed=True,
                rows=[*tier3_config_rows[3:], detail_row("verdict", tier3_gate_label, tier3_skip)],
            )
        elif tier3_ran:
            view.tier_done(tier3_index, failed=not tier3_ok, rows=[*tier3_config_rows[3:], *tier3_rows])
        else:
            view.tier_skip(tier3_index, tier3_skip)

    for result in tier1_gate_results:
        result.metadata["gating"] = {"tier": 1, "blocking": True}
    for result in tier2_gate_results:
        result.metadata["gating"] = {
            "tier": 2,
            "blocking": block_on_dedup_effective and not bool(result.metadata.get("advisory_tier2")),
        }
    if tier3_result is not None:
        source_kind = "plugin"
        if preflight_tier3_source:
            from skillevaluator.tier3.evals_spec import validate_tier3_source

            source_kind, _source_checks = validate_tier3_source(resolved_target)
        tier3_result.metadata.setdefault(
            "tier3_applicability",
            {
                "applicability": "required",
                "reason_code": "tier3_source_present",
                "source_kind": source_kind,
            },
        )
        agent_eval_payload = tier3_result.metadata.get("agent_eval")
        if isinstance(agent_eval_payload, dict):
            agent_eval_payload.setdefault("applicability", "required")
            agent_eval_payload.setdefault("reason_code", "tier3_source_present")
            agent_eval_payload.setdefault("source_kind", source_kind)
        tier3_result.metadata["gating"] = {"tier": 3, "blocking": block_on_agent_eval_effective}

    # Reporters and the exit gate consume the same finalized result objects.
    apply_policy(results, policy)
    _finalize_evaluated_source(results, evaluated_source)

    content_label = {
        CONTENT_TYPE_SKILL: "Skill",
        CONTENT_TYPE_RULES: "Rule",
        CONTENT_TYPE_WORKFLOWS: "Workflow",
        CONTENT_TYPE_PLUGIN: "Plugin",
    }.get(resolved_type, "Skill")
    # Reports and the footer name the validated content root, not the lexical "." or manifest-file argument.
    report_root = resolved_target.resolve()
    _content_relative_finding_paths(results, resolved_target, report_root)
    target_display = resolve_git_remote_url(report_root) or str(report_root)
    sarif_repository_root = resolve_git_root(report_root)
    if sarif_repository_root is None:
        sarif_repository_root = report_root if report_root.is_dir() else report_root.parent

    # Quiet mode defaults the reports to html+json (the terminal shows only
    # the summary; the files carry the findings) and points at them from the
    # footer. An EXPLICIT -r is a contract and is honored exactly — including
    # "cli", which renders the full Rich report below the pipeline view.
    effective_formats = _effective_report_formats(report_formats, quiet=quiet)
    report_basename_value = make_timestamped_basename(f"{REPORT_PREFIX}-output")
    reports_error: ReportsNotWrittenError | None = None
    try:
        emit_reports(
            results,
            report_formats=effective_formats,
            output_dir=output_dir,
            basename=report_basename_value,
            policy=policy,
            target_path=target_display,
            content_label=content_label,
            announce_paths=not quiet,
            sarif_scan_root=report_root,
            sarif_repository_root=sarif_repository_root,
        )
    except ReportsNotWrittenError as exc:
        # A requested report is missing, so the command must fail; it does so
        # after BENCHMARK.md and the footer, which no longer links that file.
        reports_error = exc
        effective_formats = tuple(fmt for fmt in effective_formats if fmt not in exc.formats)
    _record_validate_json_report(f"{report_basename_value}.json" if "json" in effective_formats else None)

    # BENCHMARK.md is generated compulsorily for skills and plugins (matches SkillEvaluator),
    # even on failure, so the publication card always reflects the latest evaluation --
    # now including Tier 3 results when --agent-eval ran. Plugin cards add component
    # coverage, Integration, and the behavior the run did not evaluate.
    if resolved_type in (CONTENT_TYPE_SKILL, CONTENT_TYPE_PLUGIN):
        from skillevaluator.reporting import BenchmarkReporter
        from skillevaluator.reporting.naming import BENCHMARK_FILENAME
        from skillevaluator.source_identity import EvaluatedSourceConflict

        output_dir.mkdir(parents=True, exist_ok=True)
        # ``_finalize_evaluated_source`` already attached and resolved the
        # identity, so a conflict aborts before any report file exists. This
        # stays as the last line of defence: the renderer re-validates the
        # carriers it is handed, and a producer can record one after the fact.
        try:
            BenchmarkReporter(
                skill_name=target_name,
                content_type="plugin" if resolved_type == CONTENT_TYPE_PLUGIN else "skill",
            ).save(results, output_dir / BENCHMARK_FILENAME)
        except EvaluatedSourceConflict as exc:
            # Publication fails closed on a contradictory identity, so report which
            # values disagreed rather than letting the card write a guess.
            raise click.ClickException(
                f"{BENCHMARK_FILENAME} was not written because the run records more than one evaluated source ({exc})."
            ) from exc

    effective_gate_results = list(tier1_gate_results)
    if block_on_dedup_effective:
        effective_gate_results.extend(
            result for result in tier2_gate_results if not (result.metadata or {}).get("advisory_tier2")
        )
    if tier3_result is not None and block_on_agent_eval_effective:
        effective_gate_results.append(tier3_result)
    gate_failed = not all(result.passed for result in effective_gate_results)
    if quiet:
        _finish_pipeline_view(
            view,
            tier_gate_results=effective_gate_results,
            tier3_result=tier3_result,
            gate_failed=gate_failed,
            output_dir=output_dir,
            basename=report_basename_value,
            report_formats=effective_formats,
            target_path=report_root,
            agent_eval=agent_eval,
        )
    if reports_error is not None:
        if gate_failed:
            raise click.ClickException(f"validation failed, and {reports_error.message}")
        raise reports_error
    if gate_failed:
        raise click.ClickException("validation failed")


# Intro text shown under the grouped Tier 3 options in ``validate --help``.
validate.help_group_descriptions = {
    _RUN_GROUP: _RUN_GROUP_DESC,
    _TIER1_GROUP: _TIER1_GROUP_DESC,
    _TIER2_GROUP: _TIER2_GROUP_DESC,
    _TIER3_GROUP: _TIER3_GROUP_DESC,
}


@cli.command("quality-check")
@_target_argument
@click.option("--min-score", type=int, default=70, show_default=True)
@_report_options
def quality_check(target_path: Path, min_score: int, report_formats: tuple[str, ...], output_dir: Path) -> None:
    """Score skill quality across correctness, discoverability, reliability, and efficiency."""
    if not emit_reports(
        run_quality_check(target_path, min_score=min_score),
        report_formats=report_formats,
        output_dir=output_dir,
        basename=report_basename("quality"),
    ):
        raise click.ClickException("quality check failed")


@cli.command("rubric-eval")
@_target_argument
@click.option("--min-score", type=int, default=70, show_default=True)
@_report_options
def rubric_eval(target_path: Path, min_score: int, report_formats: tuple[str, ...], output_dir: Path) -> None:
    """Run LLM-as-judge rubric evaluation for a skill."""
    if not emit_reports(
        run_rubric_eval(target_path, min_score=min_score),
        report_formats=report_formats,
        output_dir=output_dir,
        basename=report_basename("rubric"),
    ):
        raise click.ClickException("rubric evaluation failed")


@cli.command("security-scan")
@_target_argument
@click.option("--llm/--no-llm", default=False, show_default=True, help="Enable LLM security analysis.")
@click.option("--llm-verify", is_flag=True, help="Use LLM verification to reduce false positives.")
@_report_options
def security_scan(
    target_path: Path, llm: bool, llm_verify: bool, report_formats: tuple[str, ...], output_dir: Path
) -> None:
    """Scan for security vulnerabilities."""
    if not emit_reports(
        run_security_scan(target_path, use_llm=llm, llm_verify=llm_verify),
        report_formats=report_formats,
        output_dir=output_dir,
        basename=report_basename("security"),
    ):
        raise click.ClickException("security scan failed")


@cli.command("pii-scan")
@_target_argument
@click.option("--llm-verify", is_flag=True, help="Use LLM verification to reduce false positives.")
@_report_options
def pii_scan(target_path: Path, llm_verify: bool, report_formats: tuple[str, ...], output_dir: Path) -> None:
    """Scan for PII and local identifiers."""
    if not emit_reports(
        run_pii_scan(target_path, llm_verify=llm_verify),
        report_formats=report_formats,
        output_dir=output_dir,
        basename=report_basename("pii"),
    ):
        raise click.ClickException("PII scan failed")


@cli.command("lint-scripts")
@_target_argument
@_report_options
def lint_scripts(target_path: Path, report_formats: tuple[str, ...], output_dir: Path) -> None:
    """Run advisory lint checks on skill scripts."""
    if not emit_reports(
        run_lint_scripts(target_path),
        report_formats=report_formats,
        output_dir=output_dir,
        basename=report_basename("script-lint"),
    ):
        raise click.ClickException("script lint failed")


@cli.command("similarity-check")
@click.argument("content_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--type",
    "content_type",
    default="auto",
    type=click.Choice(["skill", "rules", "workflows", "plugin", "auto"]),
    help="Content type; plugin builds a local catalog of plugins and their bundled skills (with --save-catalog).",
)
@click.option("--threshold", type=float, default=0.75, show_default=True, callback=_validate_similarity_threshold)
@click.option("--full-body", is_flag=True, help="Embed full file bodies instead of descriptions.")
@click.option("--model", default=None, help="Embedding model override.")
@click.option(
    "--max-entries",
    type=click.IntRange(1, SIMILARITY_MAX_ENTRIES),
    default=SIMILARITY_DEFAULT_MAX_ENTRIES,
    show_default=True,
    help="Maximum selected manifests for a fresh collection scan.",
)
@click.option(
    "--max-scalar-comparisons",
    type=click.IntRange(min=1),
    default=SIMILARITY_DEFAULT_MAX_SCALAR_COMPARISONS,
    show_default=True,
    help="Maximum comparisons multiplied by embedding dimensions.",
)
@click.option(
    "--catalog",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Compare exactly one skill against a local catalog.",
)
@click.option(
    "--save-catalog",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Build and save a versioned local catalog from this collection (skills, or plugins with bundled skills).",
)
@click.option("--cache", type=click.Path(path_type=Path), default=None, hidden=True)
@click.option("--save-cache", type=click.Path(path_type=Path), default=None, hidden=True)
@_report_options
def similarity_check(
    content_path: Path,
    content_type: str,
    threshold: float,
    full_body: bool,
    model: str | None,
    max_entries: int,
    max_scalar_comparisons: int,
    catalog: Path | None,
    save_catalog: Path | None,
    cache: Path | None,
    save_cache: Path | None,
    report_formats: tuple[str, ...],
    output_dir: Path,
) -> None:
    """Detect duplicate content with embedding similarity."""
    from skillevaluator.tier2.commands import run_similarity_check

    if catalog and cache:
        raise click.UsageError("--catalog and deprecated --cache cannot be used together")
    if save_catalog and save_cache:
        raise click.UsageError("--save-catalog and deprecated --save-cache cannot be used together")
    resolved_catalog = catalog or cache
    resolved_save_catalog = save_catalog or save_cache
    if resolved_catalog and resolved_save_catalog:
        raise click.UsageError("--catalog and --save-catalog cannot be used together")

    _reject_linked_tier2_root(content_path)
    similarity_basename = report_basename("similarity")
    _reject_catalog_report_collisions(
        resolved_catalog,
        report_formats=report_formats,
        output_dir=output_dir,
        basename=similarity_basename,
    )
    _reject_catalog_report_collisions(
        resolved_save_catalog,
        report_formats=report_formats,
        output_dir=output_dir,
        basename=similarity_basename,
    )

    results = run_similarity_check(
        content_path,
        content_type=content_type,
        threshold=threshold,
        full_body=full_body,
        model=model,
        max_entries=max_entries,
        max_scalar_comparisons=max_scalar_comparisons,
        catalog=resolved_catalog,
        save_catalog=resolved_save_catalog,
    )
    sanitize_tier2_results(results, content_path, resolved_catalog, resolved_save_catalog)

    if not emit_reports(
        results,
        report_formats=report_formats,
        output_dir=output_dir,
        basename=similarity_basename,
    ):
        raise click.ClickException("similarity check failed")


@cli.command("context-optimization-check")
@_tier2_skill_argument
@click.option("--threshold", type=float, default=0.80, show_default=True, callback=_validate_similarity_threshold)
@click.option("--model", default=None, help="Embedding model override.")
@click.option("--llm-model", default=None, help="LLM model override.")
@_report_options
def context_optimization_check(
    skill_path: Path,
    threshold: float,
    model: str | None,
    llm_model: str | None,
    report_formats: tuple[str, ...],
    output_dir: Path,
) -> None:
    """Detect redundant content within one skill."""
    from skillevaluator.tier2.commands import run_context_optimization_check

    _reject_linked_tier2_root(skill_path)
    results = run_context_optimization_check(skill_path, threshold=threshold, model=model, llm_model=llm_model)
    sanitize_tier2_results(results, skill_path)
    if not emit_reports(
        results,
        report_formats=report_formats,
        output_dir=output_dir,
        basename=report_basename("context"),
    ):
        raise click.ClickException("context optimization check failed")


@cli.command("dedup-scan")
@_tier2_skill_argument
@click.option("--threshold", type=float, default=0.80, show_default=True, callback=_validate_similarity_threshold)
@click.option("--llm-model", default=None, help="LLM model override.")
@click.option("--model", default=None, help="Embedding model override.")
@_report_options
def dedup_scan(
    skill_path: Path,
    threshold: float,
    llm_model: str | None,
    model: str | None,
    report_formats: tuple[str, ...],
    output_dir: Path,
) -> None:
    """Detect semantically redundant content within one skill."""
    from skillevaluator.tier2.commands import run_dedup_scan

    _reject_linked_tier2_root(skill_path)
    results = run_dedup_scan(
        skill_path,
        threshold=threshold,
        llm_model=llm_model,
        model=model,
    )
    sanitize_tier2_results(results, skill_path)
    if not emit_reports(
        results,
        report_formats=report_formats,
        output_dir=output_dir,
        basename=report_basename("dedup"),
    ):
        raise click.ClickException("dedup scan failed")


def _workflow_report_options(func):
    func = _report_options(func)
    for param in func.__click_params__:
        if param.name == "report_formats":
            param.default = ("cli", "json", "html")
            param.help = "Report formats; repeat -r or separate values with commas."
    return func


@click.command(cls=cli.command_class, context_settings=CONTEXT_SETTINGS)
@click.argument("skill_path", metavar="PATH", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--checks", default=None, help="Select static checks; by default all checks include dependency auditing.")
@click.option(
    "--llm/--no-llm", default=None, help="Include rubric, LLM security, and verification [default: when configured]."
)
@click.option("--min-score", type=click.IntRange(0, 100), default=70, show_default=True)
@click.option("--profile", default=None, help="Validation policy profile [default: external].")
@click.option("--previous-version", default=None, help="Previous version for version checks.")
@_workflow_report_options
def _tier1_workflow(
    skill_path: Path,
    checks: str | None,
    llm: bool | None,
    min_score: int,
    profile: str | None,
    previous_version: str | None,
    report_formats: tuple[str, ...],
    output_dir: Path,
) -> None:
    """Run Tier 1 static checks and configured LLM checks for one skill."""
    from skillevaluator.tier_workflows import run_tier1_workflow
    from skillevaluator.validators.policy import resolve_policy

    try:
        policy = resolve_policy(profile=profile)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    results = run_tier1_workflow(
        skill_path,
        checks=checks,
        llm=llm,
        min_score=min_score,
        profile=policy,
        previous_version=previous_version,
    )
    if not emit_reports(
        results,
        report_formats=report_formats,
        output_dir=output_dir,
        basename=report_basename("tier1"),
        policy=policy,
        target_path=str(skill_path),
    ):
        raise click.ClickException("Tier 1 checks failed or were incomplete")


@click.command(cls=cli.command_class, context_settings=CONTEXT_SETTINGS)
@click.argument("skill_path", metavar="PATH", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--catalog",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Saved JSON catalog for inter-skill comparison; for plugins, also inter-plugin comparison.",
)
@click.option(
    "--threshold",
    type=float,
    default=0.80,
    show_default=True,
    callback=_validate_similarity_threshold,
    help="Intra-skill similarity threshold.",
)
@click.option(
    "--similarity-threshold",
    type=float,
    default=0.75,
    show_default=True,
    callback=_validate_similarity_threshold,
    help="Inter-skill and inter-plugin catalog similarity threshold.",
)
@click.option("--full-body", is_flag=True, help="Compare full SKILL.md content against a full-body catalog.")
@click.option(
    "--llm/--no-llm",
    default=False,
    show_default=True,
    help="Plugin only: add an advisory LLM verdict to inter-plugin catalog matches (requires --catalog).",
)
@_workflow_report_options
def _tier2_workflow(
    skill_path: Path,
    catalog: Path | None,
    threshold: float,
    similarity_threshold: float,
    full_body: bool,
    llm: bool,
    report_formats: tuple[str, ...],
    output_dir: Path,
) -> None:
    """Run intra-skill deduplication and optional local catalog comparison for a skill or plugin."""
    from skillevaluator.tier_workflows import run_tier2_workflow

    _reject_linked_tier2_root(skill_path)
    basename = report_basename("tier2")
    _reject_catalog_report_collisions(catalog, report_formats=report_formats, output_dir=output_dir, basename=basename)
    results = run_tier2_workflow(
        skill_path,
        catalog=catalog,
        threshold=threshold,
        similarity_threshold=similarity_threshold,
        full_body=full_body,
        llm=llm,
    )
    sanitize_tier2_results(results, skill_path, catalog)
    if not emit_reports(
        results,
        report_formats=report_formats,
        output_dir=output_dir,
        basename=basename,
        target_path=str(skill_path),
    ):
        raise click.ClickException("Tier 2 checks failed or were incomplete")


# Hidden top-level spelling of ``tier3 evaluate`` — kept working for scripts,
# but the tier namespace is the advertised name to avoid a duplicate surface.
@cli.command(hidden=True)
@_skill_argument
@click.option(
    "-a",
    "--agents",
    default=None,
    help=(
        "Comma-separated Harbor agents (claude aliases claude-code). Default follows the provider: "
        "NVIDIA Build=opencode, OpenAI=codex, Anthropic=claude-code."
    ),
)
@click.option("--env-mode", default="docker", show_default=True, type=ENV_MODE_CHOICE)
@click.option(
    "--environment-kwarg",
    "--ek",
    multiple=True,
    help="Harbor environment constructor kwarg, KEY=VALUE. Repeat for multiple values; never pass secrets.",
)
@click.option(
    "--autopilot",
    is_flag=True,
    help="Create one eval case when no dataset/task source exists, then evaluate.",
)
@click.option("--skip-baseline", is_flag=True, help="Skip without-skill baseline.")
@click.option("--n-attempts", type=int, default=None)
@click.option("--pass-threshold", type=float, default=None)
@click.option(
    "--stop-on-pass/--no-stop-on-pass",
    default=None,
    help="Stop a case's remaining attempts once one passes.",
)
@click.option("--n-concurrent", type=int, default=None)
@click.option("--max-agents", type=int, default=None)
@click.option("--model", default=None, help="Global agent model override.")
@click.option("--agent-model", multiple=True, help="Per-agent model override, AGENT=MODEL.")
@click.option("--custom-dockerfile-mode", type=click.Choice(["preserve", "rebase"]), default=None)
@click.option("--skill-workspace-mode", type=click.Choice(["isolated", "group"]), default=None)
@click.option("--include-skills", multiple=True, type=click.Path(exists=True, path_type=Path))
@click.option("--copy-repo", is_flag=True)
@click.option("--grading-mode", type=GRADING_MODE_CHOICE, default=None)
@click.option("--results-dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=None)
@click.option("--harbor-keep-jobs", is_flag=True)
@click.option(
    "--agent-runtime-preflight/--no-agent-runtime-preflight",
    default=None,
    help="Run an extra agent-only execution of the first staged task before the full matrix [default: disabled].",
)
@click.option("--timeout-multiplier", type=float, default=None)
@click.option("--override-cpus", type=int, default=None)
@click.option("--override-memory-mb", type=int, default=None)
@click.option("--override-storage-mb", type=int, default=None)
@click.option(
    "--evaluated-source-repository",
    default=None,
    # This command renders no card, so it names where it does record the value:
    # the run directory, which a later card is rendered from.
    help="Repository (owner/name) of the source tree being evaluated, persisted into the run's run_config.json.",
)
@click.option(
    "--evaluated-source-revision",
    default=None,
    help="Immutable revision of the evaluated source: a full Git object id, or a sha256/sha384/sha512 digest.",
)
@click.option(
    "--evaluator-container-revision",
    default=None,
    help="Digest-pinned evaluator image reference recorded beside the evaluated source.",
)
@click.option(
    "--progress",
    type=click.Choice(["auto", "rich", "plain", "off"]),
    default="auto",
    show_default=True,
    help="Tier 3 progress presentation (auto uses Rich on a TTY and plain lines otherwise).",
)
def evaluate(
    skill_path: Path,
    agents: str | None,
    env_mode: str,
    environment_kwarg: tuple[str, ...],
    autopilot: bool,
    skip_baseline: bool,
    n_attempts: int | None,
    pass_threshold: float | None,
    stop_on_pass: bool | None,
    n_concurrent: int | None,
    max_agents: int | None,
    model: str | None,
    agent_model: tuple[str, ...],
    custom_dockerfile_mode: str | None,
    skill_workspace_mode: str | None,
    include_skills: tuple[Path, ...],
    copy_repo: bool,
    grading_mode: str | None,
    results_dir: Path | None,
    harbor_keep_jobs: bool,
    agent_runtime_preflight: bool | None,
    timeout_multiplier: float | None,
    override_cpus: int | None,
    override_memory_mb: int | None,
    override_storage_mb: int | None,
    evaluated_source_repository: str | None,
    evaluated_source_revision: str | None,
    evaluator_container_revision: str | None,
    progress: str,
) -> None:
    """Run Tier 3 live agent evaluation."""
    from skillevaluator.evaluation import EvaluationOptions, EvaluationService
    from skillevaluator.tier3.harbor.progress import create_progress_reporter

    service = EvaluationService()
    # Resolved before the service is built so a non-canonical value is refused
    # at the boundary rather than after a long live run has already started.
    evaluated_source = _evaluated_source_from_options(
        evaluated_source_repository,
        evaluated_source_revision,
        evaluator_container_revision,
    )
    if autopilot:
        _ensure_autopilot_dataset(skill_path, progress=progress)

    options = EvaluationOptions(
        skill_path=skill_path,
        agents=agents,
        env_mode=env_mode,
        environment_kwarg=environment_kwarg,
        skip_baseline=skip_baseline,
        n_attempts=n_attempts,
        pass_threshold=pass_threshold,
        stop_on_pass=stop_on_pass,
        n_concurrent=n_concurrent,
        max_agents=max_agents,
        model=model,
        agent_model=agent_model,
        custom_dockerfile_mode=custom_dockerfile_mode,
        skill_workspace_mode=skill_workspace_mode,
        include_skills=include_skills,
        copy_repo=copy_repo,
        grading_mode=grading_mode,
        results_dir=results_dir,
        harbor_keep_jobs=harbor_keep_jobs,
        agent_runtime_preflight=agent_runtime_preflight,
        timeout_multiplier=timeout_multiplier,
        override_cpus=override_cpus,
        override_memory_mb=override_memory_mb,
        override_storage_mb=override_storage_mb,
        evaluated_source=evaluated_source,
    )
    try:
        if env_mode == "local":
            from rich.panel import Panel
            from rich.text import Text

            console.print(
                Panel(
                    Text(
                        "Intended for trusted skills and workspaces. Local execution uses host OS safeguards; "
                        "use Docker when you need stronger isolation for untrusted code.",
                        style="yellow",
                    ),
                    title=Text("Local mode · Experimental", style="bold cyan"),
                    border_style="yellow",
                    padding=(0, 1),
                )
            )
        progress_reporter = create_progress_reporter(progress, stream=click.get_text_stream("stderr"))
        engine_result = service.evaluate(options, progress_reporter=progress_reporter)
        failure = service.failure_reason(engine_result)
        display_result = engine_result
        if failure and (
            not isinstance(engine_result, dict)
            or not (engine_result.get("error") or engine_result.get("execution_errors"))
        ):
            display_result = {
                **(engine_result if isinstance(engine_result, dict) else {}),
                "execution_status": "failed",
                "execution_errors": [failure],
            }
        if isinstance(display_result, dict):
            from skillevaluator.tier3.result_display import render_evaluation_result

            render_evaluation_result(display_result, console=console)
        if failure:
            raise click.exceptions.Exit(1)
    except (click.ClickException, click.exceptions.Exit):
        raise
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command("evaluate-plugin", hidden=True)
@click.argument("plugin_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--evals-source",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Workflow evals directory, dataset file, or skill/plugin directory containing evals/.",
)
@click.option(
    "-a",
    "--agents",
    default=None,
    help=(
        "Comma-separated Harbor agents (claude aliases claude-code). Default follows the provider: "
        "NVIDIA Build=opencode, OpenAI=codex, Anthropic=claude-code."
    ),
)
@click.option("--env-mode", default="docker", show_default=True, type=ENV_MODE_CHOICE)
@click.option("--skip-baseline", is_flag=True, help="Skip without-plugin baseline.")
@click.option(
    "--lift-mode",
    type=click.Choice(["effectiveness", "integration", "both"]),
    default="effectiveness",
    show_default=True,
    help=(
        "Compare against no plugin, sum-of-parts, or both baselines. Integration "
        "requires a cross-component dataset case; 'both' falls back to effectiveness "
        "when composition evidence is unavailable."
    ),
)
@click.option(
    "--plugin-load",
    type=click.Choice(PLUGIN_LOAD_CHOICES),
    default="wrapper",
    show_default=True,
    help=(
        "How the with-plugin arm loads the plugin: 'wrapper' (generated wrapper skill), 'native' "
        "(the harness's own plugin layout; fails for unsupported agents or local mode), or 'auto' "
        "(native where supported, the wrapper otherwise). Baseline arms are unchanged."
    ),
)
@click.option("--n-attempts", type=int, default=None)
@click.option("--pass-threshold", type=float, default=None)
@click.option("--stop-on-pass/--no-stop-on-pass", default=None)
@click.option("--n-concurrent", type=int, default=None)
@click.option("--max-agents", type=int, default=None)
@click.option("--model", default=None, help="Global agent model override.")
@click.option("--agent-model", multiple=True, help="Per-agent model override, AGENT=MODEL.")
@click.option("--custom-dockerfile-mode", type=click.Choice(["preserve", "rebase"]), default=None)
@click.option("--include-skills", multiple=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--repo-root",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Clone-root override for deterministic same-repository reference resolution.",
)
@click.option("--copy-repo", is_flag=True)
@click.option(
    "--probe-mcp",
    is_flag=True,
    help=(
        "Before the run, probe each author-supplied URL MCP server from the host (initialize + tools/list, "
        "bounded) under the endpoint policy. Sends only literal declared headers unless --probe-mcp-env names "
        "a variable. Advisory; recorded as mcp_proof in plugin provenance."
    ),
)
@click.option(
    "--probe-mcp-env",
    "probe_mcp_env",
    multiple=True,
    metavar="NAME[=HOST|@SERVER]",
    callback=_validate_probe_mcp_env,
    help=_PROBE_MCP_ENV_HELP,
)
@click.option(
    "--policy",
    "policy_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Custom policy YAML overlaid on the default profile, as in validate: its severity_overrides apply to the "
        "MCP checks that gate staging, and mcp.allowed_private_hosts reaches --probe-mcp."
    ),
)
@click.option("--grading-mode", type=GRADING_MODE_CHOICE, default=None)
@click.option("--results-dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=None)
@click.option("--harbor-keep-jobs", is_flag=True)
@click.option("--agent-runtime-preflight/--no-agent-runtime-preflight", default=None)
@click.option("--timeout-multiplier", type=float, default=None)
@click.option("--override-cpus", type=int, default=None)
@click.option("--override-memory-mb", type=int, default=None)
@click.option("--override-storage-mb", type=int, default=None)
@click.option(
    "--progress",
    type=click.Choice(["auto", "rich", "plain", "off"]),
    default="auto",
    show_default=True,
)
def evaluate_plugin(
    plugin_path: Path,
    evals_source: Path | None,
    agents: str | None,
    env_mode: str,
    skip_baseline: bool,
    lift_mode: str,
    plugin_load: str,
    n_attempts: int | None,
    pass_threshold: float | None,
    stop_on_pass: bool | None,
    n_concurrent: int | None,
    max_agents: int | None,
    model: str | None,
    agent_model: tuple[str, ...],
    custom_dockerfile_mode: str | None,
    include_skills: tuple[Path, ...],
    repo_root: Path | None,
    copy_repo: bool,
    probe_mcp: bool,
    probe_mcp_env: tuple[str, ...],
    policy_path: Path | None,
    grading_mode: str | None,
    results_dir: Path | None,
    harbor_keep_jobs: bool,
    agent_runtime_preflight: bool | None,
    timeout_multiplier: float | None,
    override_cpus: int | None,
    override_memory_mb: int | None,
    override_storage_mb: int | None,
    progress: str,
) -> None:
    """Run Tier 3 live evaluation for a public agent plugin."""
    import tempfile

    from skillevaluator.cli_core import resolve_plugin_path
    from skillevaluator.evaluation import EvaluationService
    from skillevaluator.evaluation.tier3_report import incomplete_reason, refresh_plugin_run_report
    from skillevaluator.tier3.harbor.progress import create_progress_reporter
    from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package
    from skillevaluator.validators.policy import resolve_policy

    plugin_dir = resolve_plugin_path(plugin_path)
    service = EvaluationService()
    try:
        policy = resolve_policy(policy_path=policy_path)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    try:
        with tempfile.TemporaryDirectory(prefix="skillevaluator-plugin-eval-") as temp_dir:
            prepared = prepare_plugin_eval_package(
                plugin_dir,
                stage_root=Path(temp_dir),
                evals_source=evals_source,
                include_skills=include_skills,
                repo_root=repo_root,
                plugin_load=plugin_load,
                agents=agents,
                env_mode=env_mode,
                policy=policy,
            )
            for label, values in (
                ("Unresolved remote skill refs", prepared.unresolved_skill_refs),
                ("Unresolved remote rule refs", prepared.unresolved_rule_refs),
                ("Provider-only MCP servers", prepared.unresolved_mcp_servers),
            ):
                if values:
                    console.print(
                        f"[yellow]{label} (deferred, not evaluated):[/yellow] {escape_markup(', '.join(values))}"
                    )
            if prepared.skipped or prepared.package_path is None:
                if prepared.incomplete_skip:
                    # Declared components exist but none could be evaluated: INCOMPLETE, never exit 0.
                    raise click.ClickException(
                        f"{incomplete_reason(prepared.provenance())} (nothing was evaluated). {prepared.skip_reason}"
                    )
                console.print(
                    f"[yellow]Skipping plugin evaluation:[/yellow] {escape_markup(str(prepared.skip_reason))}"
                )
                return
            effective_lift_mode, integration_skip_reason = _plugin_lift_mode_for_evidence(prepared, lift_mode)
            if integration_error := _plugin_integration_error(lift_mode, skip_baseline, integration_skip_reason):
                raise click.ClickException(f"Plugin {integration_error}")
            if integration_skip_reason:
                # --lift-mode both without composition evidence.
                console.print(
                    f"[yellow]Integration skipped:[/yellow] {escape_markup(integration_skip_reason)}. "
                    "Running effectiveness only."
                )
            allowed_private_hosts = tuple(policy.mcp_allowed_private_hosts) if probe_mcp else ()
            mcp_proof = _plugin_mcp_proof(
                prepared,
                probe_mcp=probe_mcp,
                allowed_private_hosts=allowed_private_hosts,
                probe_mcp_env=probe_mcp_env,
            )
            if probe_mcp and mcp_proof:
                for server, entry in mcp_proof.items():
                    console.print(
                        f"MCP proof [bold]{escape_markup(server)}[/bold]: {escape_markup(entry['status'])} "
                        f"[dim]({escape_markup(entry['detail'])})[/dim]"
                    )

            options = _plugin_evaluation_options(
                prepared,
                plugin_dir=plugin_dir,
                lift_mode=lift_mode,
                effective_lift_mode=effective_lift_mode,
                integration_skip_reason=integration_skip_reason,
                plugin_load=plugin_load,
                results_dir=results_dir,
                agents=agents,
                env_mode=env_mode,
                skip_baseline=skip_baseline,
                n_attempts=n_attempts,
                pass_threshold=pass_threshold,
                stop_on_pass=stop_on_pass,
                n_concurrent=n_concurrent,
                max_agents=max_agents,
                model=model,
                agent_model=agent_model,
                custom_dockerfile_mode=custom_dockerfile_mode,
                copy_repo=copy_repo,
                grading_mode=grading_mode,
                harbor_keep_jobs=harbor_keep_jobs,
                agent_runtime_preflight=agent_runtime_preflight,
                timeout_multiplier=timeout_multiplier,
                override_cpus=override_cpus,
                override_memory_mb=override_memory_mb,
                override_storage_mb=override_storage_mb,
            )
            reporter = create_progress_reporter(progress, stream=click.get_text_stream("stderr"))
            engine_result = service.evaluate(options, progress_reporter=reporter)
            lift_metadata = _plugin_lift_metadata(lift_mode, effective_lift_mode, integration_skip_reason)
            failure = service.failure_reason(engine_result)
            # Build the plugin provenance (and write its sidecar) before the summary, so the summary shows the
            # same coverage, completeness, and load blocks as the reports (the engine result alone has none).
            provenance: dict[str, Any] | None = None
            provenance_error: Exception | None = None
            if failure:
                # Keep the with-plugin evidence of a run that did not complete.
                provenance = _incomplete_plugin_provenance(prepared, engine_result, mcp_proof, failure, lift_metadata)
            else:
                try:
                    provenance = _completed_plugin_provenance(prepared, engine_result, mcp_proof, lift_metadata)
                except Exception as exc:  # still show the run summary first
                    provenance_error = exc
            if isinstance(engine_result, dict):
                from skillevaluator.tier3.result_display import render_evaluation_result

                shown = engine_result if provenance is None else {**engine_result, "plugin_provenance": provenance}
                render_evaluation_result(shown, console=console)
            if provenance_error is not None:
                raise provenance_error
            run_dir = _engine_run_dir(engine_result)
            if run_dir is not None and provenance is not None:
                # The runner rendered report.html before the sidecar existed.
                refresh_plugin_run_report(
                    plugin_dir,
                    run_dir,
                    env_mode=env_mode,
                    engine_result=engine_result,
                    plugin_provenance=provenance,
                )
            if failure:
                # The INCOMPLETE reason every report gives: why the run did not complete, then what it deferred.
                fallback = {"execution_incomplete": f"Tier 3 plugin evaluation did not complete: {failure}"}
                raise click.ClickException(incomplete_reason(provenance or fallback))
            if provenance and provenance.get("partial"):
                raise click.ClickException(incomplete_reason(provenance))
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command("create-eval-dataset")
@_skill_argument
@click.option(
    "--full",
    is_flag=True,
    help="Generate the full bucket set (up to four cases; template mode omits negative without eval guidance).",
)
@click.option("--no-llm", is_flag=True, help="Use local templates only.")
@click.option("--dry-run", is_flag=True, help="Preview without writing.")
@click.option("--force", is_flag=True, help="Overwrite existing evals/evals.json.")
@click.option("--prompt", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--refine", is_flag=True, help="Refine cases using existing or collected trajectories.")
@click.option("--from-results", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--results-dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=None)
def create_dataset(
    skill_path: Path,
    full: bool,
    no_llm: bool,
    dry_run: bool,
    force: bool,
    prompt: Path | None,
    refine: bool,
    from_results: Path | None,
    results_dir: Path | None,
) -> None:
    """Create synthetic eval datasets for agent skill evaluation."""
    from skillevaluator.evaluation import DatasetGenerationError, DatasetOptions, EvaluationService

    try:
        EvaluationService().create_dataset(
            DatasetOptions(
                skill_path=skill_path,
                full=full,
                no_llm=no_llm,
                dry_run=dry_run,
                force=force,
                prompt=prompt,
                refine=refine,
                from_results=from_results,
                results_dir=results_dir,
            )
        )
    except DatasetGenerationError as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command("init-custom-grader")
@_skill_argument
@click.option("--mode", type=CUSTOM_GRADING_MODE_CHOICE, default="default_plus_custom", show_default=True)
@click.option("--language", type=click.Choice(["python", "shell"]), default="python", show_default=True)
@click.option("--force", is_flag=True, help="Overwrite an existing top-level custom grader.")
@click.option(
    "--no-config", is_flag=True, help="Only create the grader file; do not create or update evals/config.yml."
)
def init_custom_grader(skill_path: Path, mode: str, language: str, force: bool, no_config: bool) -> None:
    """Create a BYOG custom grader starter under evals/."""
    from skillevaluator.tier3.commands import init_custom_grader as tier3_init_custom_grader

    raise SystemExit(
        tier3_init_custom_grader(
            skill_path,
            mode=mode,
            language=language,
            force=force,
            no_config=no_config,
        )
    )


@cli.command("init-harbor-task")
@_skill_argument
@click.option("--force", is_flag=True, help="Overwrite an existing starter case.")
@click.option("--case-id", default="case-001", show_default=True, help="Harbor case directory and eval entry id.")
@click.option(
    "--mode",
    type=GRADING_MODE_CHOICE,
    default="custom_only",
    show_default=True,
)
@click.option("--language", type=click.Choice(["python", "shell"]), default="python", show_default=True)
@click.option("--with-config", is_flag=True, help="Create or update evals/config.yml for native Harbor mode.")
def init_harbor_task(
    skill_path: Path,
    force: bool,
    case_id: str,
    mode: str,
    language: str,
    with_config: bool,
) -> None:
    """Create a BYOT Harbor starter template under evals/harbor/."""
    from skillevaluator.tier3.commands import init_harbor_task as tier3_init_harbor_task

    raise SystemExit(
        tier3_init_harbor_task(
            skill_path,
            force=force,
            case_id=case_id,
            mode=mode,
            language=language,
            with_config=with_config,
        )
    )


@cli.command()
@_skill_argument
@click.option("--results-dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=None)
def compare(skill_path: Path, results_dir: Path | None) -> None:
    """Compare live evaluation results across agents."""
    from skillevaluator.tier3.commands import compare_results

    raise SystemExit(compare_results(skill_path, results_dir=results_dir))


@cli.command()
@_skill_argument
@click.option("--results-dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=None)
def view(skill_path: Path, results_dir: Path | None) -> None:
    """Open the latest HTML live-evaluation report."""
    from skillevaluator.tier3.commands import view_results

    try:
        view_results(skill_path, results_dir=results_dir)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command("models")
@click.option("--limit", type=click.IntRange(min=1, max=100), default=10, show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def models_command(limit: int, as_json: bool) -> None:
    """Check catalog reachability and list visible provider models."""
    from skillevaluator.model_commands import run_models_command

    raise SystemExit(run_models_command(limit=limit, as_json=as_json))


@cli.command()
@click.option(
    "-a",
    "--agents",
    default=None,
    help=(
        "Comma-separated Harbor agents (claude aliases claude-code). Default follows the provider: "
        "NVIDIA Build=opencode, OpenAI=codex, Anthropic=claude-code."
    ),
)
@click.option("--env-mode", default="docker", show_default=True, type=ENV_MODE_CHOICE, metavar="MODE")
@click.option(
    "--environment-kwarg",
    "--ek",
    multiple=True,
    help="Harbor environment constructor kwarg, KEY=VALUE. Repeat for multiple values; never pass secrets.",
)
@click.option("--agent-model", multiple=True, help="Per-agent model override, AGENT=MODEL.")
@click.option(
    "--verify-models",
    is_flag=True,
    help="Check resolved agent-model catalog reachability with a live credential-bearing request.",
)
def doctor(
    agents: str | None,
    env_mode: str,
    environment_kwarg: tuple[str, ...],
    agent_model: tuple[str, ...],
    verify_models: bool,
) -> None:
    """Check live-evaluation runtime readiness."""
    from skillevaluator.tier3.commands import doctor as tier3_doctor

    raise SystemExit(
        tier3_doctor(
            agents=agents,
            env_mode=env_mode,
            environment_kwarg=environment_kwarg,
            verify_models=verify_models,
            agent_model=agent_model,
        )
    )


@cli.command("health-check")
@click.option(
    "-a",
    "--agents",
    default=None,
    help="Agent list; default follows the configured provider.",
)
@click.option("--env-mode", default="docker", show_default=True, type=ENV_MODE_CHOICE, metavar="MODE")
@click.option(
    "--environment-kwarg",
    "--ek",
    multiple=True,
    help="Harbor environment constructor kwarg, KEY=VALUE. Repeat for multiple values; never pass secrets.",
)
def health_check(agents: str | None, env_mode: str, environment_kwarg: tuple[str, ...]) -> None:
    """Quick readiness check for the CLI and selected live-eval backend."""
    from skillevaluator.tier3.commands import doctor as tier3_doctor

    raise SystemExit(
        tier3_doctor(
            agents=agents,
            env_mode=env_mode,
            environment_kwarg=environment_kwarg,
            verify_models=False,
            agent_model=(),
        )
    )


@tier3.command("validate")
@_skill_argument
@click.option("--json", "as_json", is_flag=True, help="Emit JSON output.")
@click.option("--strict", is_flag=True, help="Treat warnings as failures.")
@click.option("--harbor-contract", is_flag=True, help="Validate Harbor task and reward contract.")
def tier3_validate(skill_path: Path, as_json: bool, strict: bool, harbor_contract: bool) -> None:
    """Validate Tier 3 evals/ and optional Harbor BYOT contract."""
    from skillevaluator.tier3.commands import validate_evals as tier3_validate_evals

    raise SystemExit(tier3_validate_evals(skill_path, as_json=as_json, strict=strict, harbor_contract=harbor_contract))


@tier3.command("harbor-view")
@click.argument("jobs_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
def harbor_view(jobs_dir: Path) -> None:
    """Open retained Harbor job artifacts with Harbor's trajectory browser."""
    from skillevaluator.tier3.commands import harbor_view as tier3_harbor_view

    raise SystemExit(tier3_harbor_view(jobs_dir))


# Expert aliases that intentionally share the same command implementations.
tier1.workflow = _tier1_workflow
tier1.add_command(copy.copy(_tier1_workflow), "validate")
tier1.add_command(quality_check, "quality-check")
tier1.add_command(rubric_eval, "rubric-eval")
tier1.add_command(security_scan, "security-scan")
tier1.add_command(pii_scan, "pii-scan")
tier1.add_command(lint_scripts, "lint-scripts")

tier2.add_command(similarity_check, "similarity-check")
tier2.add_command(context_optimization_check, "context-optimization-check")
tier2.add_command(dedup_scan, "dedup-scan")
tier2.workflow = _tier2_workflow

# Register an independent command object for the namespaced spelling so future
# Click metadata changes on one help surface cannot leak into the other.
_tier3_evaluate_visible = copy.copy(evaluate)
# The shallow copy shares the mutable params list; give the namespaced twin its
# own list so in-place registration on one can never leak into the other.
_tier3_evaluate_visible.params = list(evaluate.params)
_tier3_evaluate_visible.hidden = False
tier3.add_command(_tier3_evaluate_visible, "evaluate")
_tier3_evaluate_plugin_visible = copy.copy(evaluate_plugin)
_tier3_evaluate_plugin_visible.params = list(evaluate_plugin.params)
_tier3_evaluate_plugin_visible.hidden = False
tier3.add_command(_tier3_evaluate_plugin_visible, "evaluate-plugin")
tier3.add_command(create_dataset, "create-eval-dataset")
tier3.add_command(init_custom_grader, "init-custom-grader")
tier3.add_command(init_harbor_task, "init-harbor-task")
tier3.add_command(compare, "compare")
tier3.add_command(view, "view")
tier3.add_command(doctor, "doctor")

from skillevaluator.tier3.workflow import build_tier3_workflow

tier3.workflow = build_tier3_workflow(evaluate)
cli.add_command(harbor_view, "harbor-view")


if __name__ == "__main__":
    cli()
