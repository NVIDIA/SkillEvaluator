# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line interface for SkillEvaluator."""

from __future__ import annotations

import copy
import json
import logging
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from skillevaluator import __version__
from skillevaluator.cli_help import GroupedOption, RichGroup
from skillevaluator.config import load_pii_patterns
from skillevaluator.constants import (
    QUALITY_DEFAULT_MIN_SCORE,
    SIMILARITY_DEFAULT_MAX_ENTRIES,
    SIMILARITY_DEFAULT_MAX_SCALAR_COMPARISONS,
    SIMILARITY_MAX_ENTRIES,
)
from skillevaluator.doctor import DEFAULT_PROBE_TIMEOUT_SEC, MAX_PROBE_TIMEOUT_SEC
from skillevaluator.harness.report import harness_block, harness_note, harness_panel, write_json_block
from skillevaluator.logging_config import setup_logging
from skillevaluator.models.result import ValidationResult
from skillevaluator.quality_config import QualityConfigError, load_quality_config
from skillevaluator.reporting.console_ui import (
    TierRow,
    ValidateView,
    ViewProgressReporter,
    check_ticker_row,
    detail_row,
    engine_feed_rows,
    sanitize_tier3_console_text,
    stage_hint_row,
    summarize_tier1,
    summarize_tier2,
    summarize_tier3,
)
from skillevaluator.reporting.naming import report_basename

# Tier 1 (static validation) is the base install surface and is safe to import
# eagerly. Tier 2 (embeddings/LLM) and Tier 3 (Harbor and its environments)
# pull heavy, extras-only dependencies, so their command implementations are
# imported lazily inside the command callbacks. This keeps `import skillevaluator.cli`
# and the CLI surface available on a base install without those extras.
from skillevaluator.tier1.commands import (
    RECOGNIZED_CHECKS,
    console,
    emit_reports,
    enabled_check_lineup,
    run_lint_scripts,
    run_pii_scan,
    run_quality_check,
    run_rubric_eval,
    run_security_scan,
    run_validation,
    runs_schema_check,
)
from skillevaluator.tier3_environments import all_env_modes
from skillevaluator.tier_group import TierGroup
from skillevaluator.utils.json_output import json_needs_ascii
from skillevaluator.utils.path_security import resolve_repo_context_root
from skillevaluator.utils.rich_markup import escape_markup, strip_terminal_controls
from skillevaluator.utils.tier2_paths import (
    is_link_or_reparse,
    paths_refer_to_same_location,
    sanitize_tier2_results,
)
from skillevaluator.validators.pii_waivers import (
    PIIWaiverError,
    configured_pii_categories,
    load_pii_waivers,
    today_utc,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from skillevaluator.reporting.presentation import FinalRenderContext
    from skillevaluator.validators.policy import ValidationPolicy


CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


ENV_MODE_CHOICE = click.Choice(list(all_env_modes()))


class _FiniteFloatRange(click.FloatRange):
    """Click float range that also rejects NaN and infinities."""

    def convert(self, value, param, ctx):
        converted = super().convert(value, param, ctx)
        if not math.isfinite(converted):
            self.fail(f"{value!r} is not a finite number", param, ctx)
        return converted


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


class _QualityConfigPath(click.Path):
    """A ``--quality-config`` file, validated while arguments are parsed."""

    def __init__(self) -> None:
        super().__init__(exists=True, dir_okay=False, path_type=Path)

    def convert(self, value, param, ctx):
        path = super().convert(value, param, ctx)
        try:
            load_quality_config(path)
        except QualityConfigError as exc:
            self.fail(str(exc), param, ctx)
        return path


QUALITY_CONFIG_PATH = _QualityConfigPath()
# Resolved so parallel catalog children receive the same absolute file.
SKILLSPECTOR_BASELINE_PATH = click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path)
_SKILLSPECTOR_BASELINE_HELP = (
    "Apply an audited SkillSpector YAML/JSON suppression baseline. Never discovered from the scanned skill."
)


class _PIIWaiversPath(click.Path):
    """A ``--pii-waivers`` file, validated while arguments are parsed."""

    def __init__(self) -> None:
        super().__init__(exists=True, dir_okay=False, path_type=Path)

    def convert(self, value, param, ctx):
        path = super().convert(value, param, ctx)
        try:
            load_pii_waivers(path, configured_pii_categories(load_pii_patterns()), today=today_utc())
        except PIIWaiverError as exc:
            self.fail(f"{path}: {exc}", param, ctx)
        return path


PII_WAIVERS_PATH = _PIIWaiversPath()
_PII_WAIVERS_HELP = "Apply reviewed, expiring PII waivers from this YAML file to the PII check."


def _validate_similarity_threshold(_ctx: click.Context, _param: click.Parameter, value: float) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise click.BadParameter("must be finite and within [0, 1]")
    return value


def _validate_progress_interval(_ctx: click.Context, _param: click.Parameter, value: float | None) -> float | None:
    if value is None:
        return None
    from skillevaluator.tier3.harbor.progress import MAX_REFRESH_INTERVAL_SECONDS, MIN_REFRESH_INTERVAL_SECONDS

    # NaN passes every range comparison, so check finiteness explicitly.
    if not math.isfinite(value) or not MIN_REFRESH_INTERVAL_SECONDS <= value <= MAX_REFRESH_INTERVAL_SECONDS:
        raise click.BadParameter(
            f"must be finite and within [{MIN_REFRESH_INTERVAL_SECONDS:g}, {MAX_REFRESH_INTERVAL_SECONDS:g}] seconds"
        )
    return value


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
             contained plugin (.claude-plugin/plugin.json)

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
  skillevaluator validate ./my-skill --profile default      # standard validation policy
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
    ("Core workflows", ("validate", "health-check", "doctor", "models", "dashboard")),
    ("Repository CI", ("ci",)),
    (
        "Tier 1 · Static and security",
        ("quality-check", "rubric-eval", "security-scan", "pii-scan", "lint-scripts", "harness-check"),
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


class _EncodingSafeGroup(RichGroup):
    """Configure output before Click parses eager options such as --help."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(errors="replace")
        return super().parse_args(ctx, args)


@click.group(
    cls=_EncodingSafeGroup,
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


@tier3.command("import-traces")
@click.argument("skill_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("export", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    help="Review bundle directory outside the skill (default: ./<skill-dir-name>-trace-review).",
)
@click.option("--force", is_flag=True, help="Overwrite an existing candidates file.")
@click.option("--dry-run", is_flag=True, help="Print the summary without writing files.")
def tier3_import_traces(skill_path: Path, export: Path, output: Path | None, force: bool, dry_run: bool) -> None:
    """Import a production trace export into redacted review candidates."""
    from skillevaluator.tier3.trace_import import import_traces

    output = output or Path.cwd() / f"{skill_path.resolve().name}-trace-review"
    try:
        document = import_traces(skill_path, export, output, force=force, dry_run=dry_run)
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"candidate: {len(document['candidates'])}")
    for reason in ("duplicate", "incomplete", "unsupported", "other skill"):
        click.echo(f"{reason}: {sum(s['reason'] == reason for s in document['skipped'])}")
    for item in document["skipped"]:
        click.echo(json.dumps(item, ensure_ascii=False))
    for candidate in document["candidates"]:
        click.echo(f"{candidate['id']}: redactions={candidate['redactions']} warnings={candidate['warnings']}")
    click.echo(
        f"Next: review {output / 'candidates.json'}, then run skillevaluator tier3 accept-traces {skill_path} {output}"
    )


@tier3.command("accept-traces")
@click.argument("skill_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("review_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--dry-run", is_flag=True, help="Report additions without writing files.")
@click.option(
    "--allow-native-harbor",
    is_flag=True,
    help="Explicitly accept traces when native Harbor tasks are selected.",
)
def tier3_accept_traces(skill_path: Path, review_dir: Path, dry_run: bool, allow_native_harbor: bool) -> None:
    """Promote only human-reviewed trace candidates into evals/evals.json."""
    from skillevaluator.tier3.trace_import import accept_traces

    try:
        result = accept_traces(skill_path, review_dir, dry_run=dry_run, allow_native_harbor=allow_native_harbor)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise click.ClickException(str(exc)) from exc
    for key in ("added", "already present", "skipped"):
        click.echo(f"{'would add' if dry_run and key == 'added' else key}: {len(result[key])} {', '.join(result[key])}")
    for check in result["validation"]:
        click.echo(str(check))
    if any(check.status == "error" for check in result["validation"]):
        raise click.ClickException("Dataset validation failed after writing")


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


def _plugin_run_options(func):
    """Reuse the same plugin controls for combined and standalone runs."""
    options = (
        (
            ("--plugin-load",),
            {
                "type": click.Choice(["wrapper", "native", "auto"]),
                "help": "Plugin loading mode [default: wrapper or evals config].",
            },
        ),
        (
            ("--lift-mode",),
            {
                "type": click.Choice(["effectiveness", "integration", "both", "ladder"]),
                "help": "Plugin comparison arms [default: effectiveness or evals config].",
            },
        ),
        (("--probe-mcp/--no-probe-mcp",), {"help": "Opt into policy-checked MCP host probes."}),
        (
            ("--probe-mcp-env",),
            {"multiple": True, "help": "Explicit environment grant for an MCP host probe (repeatable)."},
        ),
        (
            ("--trial-retries",),
            {"type": click.IntRange(0, 5), "help": "Retry setup-only failures, 0-5 [default: 0 or evals config]."},
        ),
    )
    for names, settings in reversed(options):
        settings.setdefault("default", () if settings.get("multiple") else None)
        func = click.option(*names, cls=GroupedOption, help_group=_TIER3_GROUP, **settings)(func)
    return func


def _plugin_static_options(func):
    func = click.option(
        "--repo-root",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        default=None,
        help="Repository context for same-repository plugin references.",
    )(func)
    func = click.option(
        "--resolve-endpoints", is_flag=True, help="Opt into bounded plugin endpoint DNS/redirect checks."
    )(func)
    return click.option(
        "--claude-validate", is_flag=True, help="Compare the manifest with installed Claude Code validation (advisory)."
    )(func)


def _path_is_within(candidate: Path, root: Path) -> bool:
    """Check containment using both path text and existing filesystem aliases."""
    for path, path_root in (
        (candidate.absolute(), root.absolute()),
        (candidate.resolve(strict=False), root.resolve(strict=True)),
    ):
        try:
            path.relative_to(path_root)
            return True
        except ValueError:
            pass

    current = candidate.absolute()
    while True:
        try:
            current.lstat()
            break
        except FileNotFoundError:
            parent = current.parent
            if parent == current:
                return False
            current = parent
        except OSError:
            return False

    while True:
        try:
            if current.samefile(root):
                return True
        except (OSError, ValueError):
            return False
        parent = current.parent
        if parent == current:
            return False
        current = parent


def _default_report_output_error(directory: Path, cause: BaseException) -> click.ClickException:
    """Say why the default report directory was refused and how to choose another."""
    reason = str(cause).removesuffix(f": {directory}").strip() or type(cause).__name__
    return click.ClickException(
        f"Cannot safely reserve default report output directory: {directory} ({reason}). "
        "Choose a report directory with -o/--output-dir."
    )


def _resolve_report_output_location(target_path: Path, output_dir: Path) -> Path:
    """Keep report writes outside the source tree whose identity they describe."""
    if not target_path.is_dir():
        return output_dir
    try:
        lexical_target = target_path.absolute()
        resolved_target = target_path.resolve(strict=True)
        lexical_output = output_dir.absolute()
        resolved_output = output_dir.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise click.ClickException(f"Cannot resolve report output directory: {output_dir}") from exc

    output_is_in_target = _path_is_within(lexical_output, lexical_target) or _path_is_within(
        resolved_output,
        resolved_target,
    )
    from click.core import ParameterSource

    context = click.get_current_context(silent=True)
    parameter_source = context.get_parameter_source("output_dir") if context is not None else None
    output_is_default = parameter_source is ParameterSource.DEFAULT
    catalog_root = context.meta.get("skillevaluator_catalog_report_root") if context is not None else None
    catalog_child_output = False
    if isinstance(catalog_root, Path) and not output_is_in_target:
        try:
            output_dir.absolute().relative_to(catalog_root.absolute())
            catalog_child_output = True
        except ValueError:
            if catalog_root.exists():
                catalog_child_output = _path_is_within(output_dir, catalog_root)
    if not output_is_in_target and (not output_is_default or catalog_child_output):
        return output_dir

    if not output_is_in_target:
        from skillevaluator.tier3.output_provenance import (
            is_generated_output_root,
            mark_generated_output_root,
        )

        try:
            output_is_real_directory = output_dir.is_dir() and not is_link_or_reparse(output_dir)
            output_is_empty = output_is_real_directory and next(output_dir.iterdir(), None) is None
            if (not output_dir.exists() and not is_link_or_reparse(output_dir)) or output_is_empty:
                mark_generated_output_root(output_dir)
                return output_dir
            if output_is_real_directory and is_generated_output_root(output_dir):
                return output_dir
        except (OSError, RuntimeError, ValueError) as exc:
            raise _default_report_output_error(output_dir, exc) from exc

    if output_is_default:
        relocated = target_path.with_name(f"{target_path.name}-reports")
        try:
            if not _path_is_within(relocated, resolved_target):
                from skillevaluator.tier3.output_provenance import mark_generated_output_root

                mark_generated_output_root(relocated)
                return relocated
        except (OSError, RuntimeError, ValueError) as exc:
            raise _default_report_output_error(relocated, exc) from exc
    raise click.ClickException(
        f"Report output must be outside the publication target; use an external --output-dir: {output_dir}"
    )


def _resolve_file_report_output_location(
    target_path: Path,
    output_dir: Path,
    report_formats: tuple[str, ...],
) -> Path:
    """Resolve and reserve report storage only when a file reporter needs it."""
    if not any(report_format in _FILE_REPORT_EXTENSIONS for report_format in report_formats):
        return output_dir
    return _resolve_report_output_location(target_path, output_dir)


def _reject_copy_repo_root_output(
    target_path: Path,
    output_dir: Path,
    *,
    copy_repo: bool,
    agent_eval: bool,
    include_skills: tuple[Path, ...] = (),
) -> None:
    """Reject or reserve report storage inside any Tier 3 repo context."""
    if not agent_eval:
        return
    mode_label = "--copy-repo" if copy_repo else "Tier 3 linked-context staging"
    try:
        context_roots = [(resolve_repo_context_root(target_path), "repository root")]
        context_roots.extend(
            (included_skill.resolve(strict=True), "runtime source root") for included_skill in include_skills
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(f"Cannot validate {mode_label} report output directory: {output_dir}") from exc
    overlaps_context = False
    seen_roots: set[Path] = set()
    for context_root, root_label in context_roots:
        if context_root in seen_roots:
            continue
        seen_roots.add(context_root)
        try:
            output_is_context_root = paths_refer_to_same_location(output_dir, context_root)
            output_is_in_context = _path_is_within(output_dir, context_root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise click.ClickException(f"Cannot validate {mode_label} report output directory: {output_dir}") from exc
        if output_is_context_root:
            raise click.UsageError(
                f"With {mode_label}, report output cannot be the {root_label}; "
                f"choose a dedicated path such as ./reports instead of: {output_dir}"
            )
        overlaps_context = overlaps_context or output_is_in_context
    if not overlaps_context:
        return

    # Tier 3 staging excludes this whole directory so prior reports cannot
    # influence the agent. Linked-context staging can follow authored files
    # anywhere in the repository, so it needs the same ownership check as a
    # full-repository copy before omitting a subtree wholesale.
    from skillevaluator.tier3.output_provenance import mark_generated_output_root

    try:
        mark_generated_output_root(output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.UsageError(
            f"With {mode_label}, an in-repository report output must be a dedicated "
            "SkillEvaluator generated-output directory; choose a new or authenticated "
            f"report directory instead of: {output_dir}"
        ) from exc


_validate_json_report_var: ContextVar[str | None] = ContextVar(
    "_validate_json_report_var",
    default=None,
)


def _effective_report_formats(report_formats: tuple[str, ...], *, quiet: bool) -> tuple[str, ...]:
    """Resolve the report formats ``validate`` will actually emit."""
    if quiet and not _report_formats_explicit():
        return tuple(dict.fromkeys([fmt for fmt in report_formats if fmt != "cli"] + ["html", "json"]))
    return report_formats


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


def _run_dedup_or_skip(target_path: Path) -> list[ValidationResult]:
    """Run Tier 2 dedup when possible, else return a non-failing skipped result.

    Dedup is on by default for ``validate`` but needs the ``tier2`` extra and an
    configured public embedding provider. When either is missing it degrades
    gracefully to a warning so a lightweight ``validate`` keeps working.
    """
    import importlib.util

    # Plugins share this setup guard; run_dedup_scan routes them to plugin dedup.
    def _skip(message: str) -> list[ValidationResult]:
        from skillevaluator.publication_identity import stamp_publication_target

        result = ValidationResult(
            validator_name="Tier 2 Deduplication",
            validator_description="Embedding-based duplicate detection",
        )
        result.add_warning(message)
        result.metadata["skipped"] = True
        stamp_publication_target([result], target_path)
        return [result]

    def _available(module: str) -> bool:
        # find_spec does not import the module (so nothing leaks into a base
        # install) and may raise if a meta-path blocker is active; treat any
        # failure as "unavailable" so dedup degrades to a warning.
        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            return False

    if not _available("openai"):
        return _skip("Skipped: install the Tier 2 extra (make install EXTRAS=tier2), or pass --no-dedup.")
    try:
        from skillevaluator.provider_config import ProviderConfigurationError, resolve_embedding_provider
        from skillevaluator.tier2.commands import run_dedup_scan
    except ImportError:
        return _skip("Skipped: install the Tier 2 extra (make install EXTRAS=tier2), or pass --no-dedup.")
    try:
        resolve_embedding_provider()
    except ProviderConfigurationError as exc:
        return _skip(f"Tier 2 skipped: embedding setup is required.\n\n{exc}\n\nTo skip Tier 2, pass --no-dedup.")
    return run_dedup_scan(target_path)


def _partial_agent_eval_result(
    target_path: Path,
    *,
    engine_result: object,
    failure: str,
    results_dir: Path | None,
    env_mode: str,
    plugin_run: bool = False,
) -> ValidationResult | None:
    """Normalize an engine run that produced usable results alongside errors.

    A run where one agent crashed but another scored is still a run: discarding
    it as a skip throws away real results. Tier 3 stays advisory -- the errors
    are carried on the result, rendered red in the pipeline view and combined
    report, but never gate the exit code. Returns ``None`` unless the engine
    mapping can be proven to be THIS run's fresh output (in-memory agents data
    plus a ``latest`` results dir matching the engine's ``run_dir``), so a
    stale earlier run is never reported as fresh. Plugin runs instead normalize
    the exact returned directory, preserving partial scores when latest moves.
    """
    from skillevaluator.evaluation.tier3_report import agent_eval_result_from_run
    from skillevaluator.tier3.results_location import resolve_latest_results

    if not isinstance(engine_result, dict) or not engine_result.get("agents"):
        return None
    run_dir_value = engine_result.get("run_dir")
    if not run_dir_value or not Path(str(run_dir_value)).is_dir():
        return None
    try:
        if plugin_run:
            from skillevaluator.cli_plugin import plugin_result_from_execution

            result = plugin_result_from_execution(target_path, engine_result, env_mode=env_mode)
        else:
            latest = resolve_latest_results(target_path, results_dir)
            latest = latest.resolve() if latest.is_symlink() else latest
            if latest.resolve() != Path(str(run_dir_value)).resolve():
                return None
            result = agent_eval_result_from_run(
                target_path,
                results_dir=results_dir,
                env_mode=env_mode,
                engine_result=engine_result,
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
        if message not in result.errors:
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


def _validate_skip(_ctx: click.Context, _param: click.Parameter, value: str | None) -> str | None:
    """Reject ``--skip`` values that name no known Tier 1 check."""
    from skillevaluator.tier1.commands import SKIPPABLE_CHECKS, skipped_checks

    if value is None:
        return None
    names = skipped_checks(value)
    if not names or names - SKIPPABLE_CHECKS:
        unknown = ", ".join(sorted(names - SKIPPABLE_CHECKS)) or repr(value)
        raise click.BadParameter(f"unknown check {unknown}; choose from {', '.join(sorted(SKIPPABLE_CHECKS))}")
    return value


def _offer_scanner_install(checks: str | None = None, skip: str | None = None) -> None:
    """Before Tier 1, install the pinned, checksum-verified SkillSpector and Gitleaks when missing."""
    from skillevaluator.tier1.commands import runs_secrets_scan, runs_security_scan
    from skillevaluator.utils.scanner_install import (
        GITLEAKS_VERSION,
        SKILLSPECTOR_VERSION,
        ensure_gitleaks,
        managed_skillspector_wheel,
        prepare_skillspector,
    )
    from skillevaluator.utils.tool_runner import Tools

    steps = []
    wheel = managed_skillspector_wheel()
    if (
        runs_security_scan(checks, skip)
        and Tools.skillspector.resolution_source == "uv tool run"
        and wheel is not None
        and not wheel.is_file()
    ):
        uv = Tools.skillspector.path
        steps.append((f"SkillSpector {SKILLSPECTOR_VERSION}", lambda: prepare_skillspector(uv)))
    path = Tools.gitleaks.path
    if (
        runs_secrets_scan(checks, skip)
        and Tools.gitleaks.resolution_source == "managed"
        and path is not None
        and not Path(path).is_file()
    ):
        steps.append((f"Gitleaks {GITLEAKS_VERSION}", ensure_gitleaks))
    for label, install in steps:
        with console.status(f"Installing {label}…"):
            error = install()
        if error:
            console.print(f"[red]✗ {error[:1].upper()}{error[1:]}[/red]", highlight=False)
        else:
            console.print(f"[dim]✓ {label} installed (checksum-verified)[/dim]", highlight=False)


def _run_agent_eval_or_skip(
    target_path: Path,
    *,
    agents: str | None,
    env_mode: str,
    environment_kwarg: tuple[str, ...] = (),
    skip_baseline: bool,
    n_concurrent: int | None,
    max_agents: int | None,
    max_cases: int | None = None,
    n_attempts: int | None = None,
    max_agent_tokens: int | None = None,
    max_judge_calls: int | None = None,
    max_attempts_total: int | None = None,
    max_cost_usd: float | None = None,
    price_file: Path | None = None,
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
    tolerate_baseline_timeouts: bool | None = None,
    block_on_agent_eval: bool = False,
    validate_source: bool = True,
    repo_context_exclude_paths: tuple[Path, ...] = (),
    evaluated_source: dict[str, str] | None = None,
    progress_reporter=None,
    plugin_options=None,
) -> ValidationResult:
    """Run Tier 3 live agent evaluation and fold the result into the combined report.

    Returns an ``AGENT_EVAL`` :class:`ValidationResult` carrying the canonical
    ``metadata["agent_eval"]`` payload on success, or a structured result
    describing why Tier 3 could not run. Tier 3 remains advisory by default,
    and callers can opt into blocking behavior.
    """
    from skillevaluator.publication_identity import publication_target_name

    # Persist the entry name that publication targets record, not a lexical
    # alias that a case- or normalization-insensitive filesystem accepted.
    source_path = target_path
    if plugin_options is not None:
        from skillevaluator.cli_core import resolve_plugin_path
        from skillevaluator.tier3.results_location import plugin_results_directory

        target_path = resolve_plugin_path(target_path)
        results_dir = plugin_results_directory(target_path, results_dir)
    skill_name = publication_target_name(target_path) or target_path.name
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
                skill_name=skill_name,
                source_kind=source_kind,
            )

    from skillevaluator.evaluation import EvaluationOptions, EvaluationService
    from skillevaluator.evaluation.tier3_report import (
        advisory_skip_result,
        agent_eval_result_from_run,
    )

    options = EvaluationOptions(
        skill_path=source_path if plugin_options is not None else target_path,
        agents=agents,
        env_mode=env_mode,
        environment_kwarg=environment_kwarg,
        skip_baseline=skip_baseline,
        n_concurrent=n_concurrent,
        max_agents=max_agents,
        max_cases=max_cases,
        n_attempts=n_attempts,
        max_agent_tokens=max_agent_tokens,
        max_judge_calls=max_judge_calls,
        max_attempts_total=max_attempts_total,
        max_cost_usd=max_cost_usd,
        price_file=price_file,
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
        tolerate_baseline_timeouts=tolerate_baseline_timeouts,
        evaluated_source=evaluated_source,
        repo_context_exclude_paths=repo_context_exclude_paths,
    )
    try:
        service = EvaluationService()
        if plugin_options is not None:
            engine_result = service.evaluate_plugin(
                options, plugin_options=plugin_options, progress_reporter=progress_reporter
            )
        elif progress_reporter is not None:
            engine_result = service.evaluate(options, progress_reporter=progress_reporter)
        else:
            engine_result = service.evaluate(options)
    except Exception as exc:
        # Preserve a reportable Tier 3 result rather than aborting after the
        # earlier tiers have already completed. The caller's gating metadata
        # decides whether this failure is advisory or blocking.
        return advisory_skip_result(
            f"Tier 3 live evaluation skipped: {exc}",
            skill_name=skill_name,
        )

    if failure := service.failure_reason(engine_result):
        partial = _partial_agent_eval_result(
            target_path,
            engine_result=engine_result,
            failure=failure,
            results_dir=results_dir,
            env_mode=env_mode,
            plugin_run=plugin_options is not None,
        )
        if partial is not None:
            return _attach_plugin_report(partial, engine_result)
        return _attach_plugin_report(
            advisory_skip_result(
                f"Tier 3 live evaluation did not complete: {failure}",
                skill_name=skill_name,
            ),
            engine_result,
        )

    try:
        if plugin_options is not None:
            from skillevaluator.cli_plugin import plugin_result_from_execution

            result = plugin_result_from_execution(target_path, engine_result, env_mode=env_mode)
        else:
            result = agent_eval_result_from_run(
                target_path,
                results_dir=results_dir,
                env_mode=env_mode,
                engine_result=engine_result if isinstance(engine_result, dict) else None,
            )
    except Exception as exc:
        return _attach_plugin_report(
            advisory_skip_result(
                f"Tier 3 result normalization failed: {exc}",
                skill_name=skill_name,
            ),
            engine_result,
        )
    if result is None:
        return _attach_plugin_report(
            advisory_skip_result(
                "Tier 3 live evaluation produced no parseable results.",
                skill_name=skill_name,
            ),
            engine_result,
        )
    return _attach_plugin_report(result, engine_result)


def _attach_plugin_report(result: ValidationResult, engine_result: object) -> ValidationResult:
    """Keep passive plugin evidence even when canonical score normalization fails."""
    if isinstance(engine_result, dict):
        if "plugin_runtime_report" in engine_result:
            result.metadata["plugin_runtime_report"] = engine_result["plugin_runtime_report"]
        if isinstance(engine_result.get("plugin_inventory"), dict):
            result.metadata.setdefault("plugin", engine_result["plugin_inventory"])
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


def _offer_local_runtime_install(raw_agents: str | None) -> None:
    """Before a local-mode run, offer to install each missing agent CLI (pinned, managed root)."""
    import shlex
    import sys

    from skillevaluator.provider_config import ProviderConfigurationError, resolve_llm_provider
    from skillevaluator.tier3.commands import resolve_agents
    from skillevaluator.tier3.harbor.local_runtime import (
        install_local_runtime,
        managed_install_argv,
        managed_runtime_prefix,
        missing_local_runtimes,
    )

    # Without a provider Tier 3 cannot run, so there is nothing to install yet; Tier 3 reports why.
    try:
        agents = resolve_agents(raw_agents, provider=resolve_llm_provider().provider)
    except (ProviderConfigurationError, ValueError):
        return
    for agent in missing_local_runtimes(agents):
        argv = managed_install_argv(agent)
        console.print(f"[yellow]Local mode needs the {agent} CLI, and it is not installed.[/yellow]")
        console.print(f"  [dim]{shlex.join(argv)}[/dim]", highlight=False, soft_wrap=True)
        if not sys.stdin.isatty():
            console.print("  [dim]Run the command above, or use --env-mode docker.[/dim]")
            continue
        if not click.confirm(f"  Install it into {managed_runtime_prefix(agent)}?", default=True):
            continue
        with console.status(f"Installing {agent}…"):
            error = install_local_runtime(agent)
        if error:
            console.print(f"  [red]✗ {error}[/red]", highlight=False)
        else:
            console.print(f"  [green]✓ {agent} installed[/green]")


def _ensure_autopilot_dataset(
    skill_path: Path,
    *,
    quiet: bool = False,
    progress: str = "auto",
    evaluation_params: dict[str, Any] | None = None,
    prepare_environment: Callable[[], None] | None = None,
) -> str | None:
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
    from skillevaluator.tier3.dataset_coverage import starter_guidance
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
            if evaluation_params is not None and evaluation_params.get("backend") == "nemo-gym":
                from skillevaluator.tier3.workflow import _source_snapshot, nemo_gym_preflight_errors

                params = {**evaluation_params, "skill_path": skill_path}
                with _source_snapshot(skill_path) as source_path:
                    errors = nemo_gym_preflight_errors(source_path, params)
                if errors:
                    raise click.UsageError("; ".join(errors)) from None
            echo("Autopilot: no public provider key is configured; generating one deterministic case.")
        else:
            no_llm = False
            if evaluation_params is not None:
                from skillevaluator.tier3.workflow import (
                    _preflight_environment,
                    _preflight_nemo_gym,
                    _preflight_options,
                    _source_snapshot,
                )

                # Match the direct Tier 3 workflow before spending a generation
                # request. Keep keyless deterministic starters available, and
                # leave existing authored sources untouched above.
                params = {**evaluation_params, "skill_path": skill_path}
                with _source_snapshot(skill_path) as source_path:
                    if params.get("backend") == "nemo-gym":
                        _preflight_nemo_gym(source_path, params)
                    else:
                        environment_kwargs = _preflight_options(source_path, params)
                if params.get("backend") != "nemo-gym":
                    if prepare_environment is not None:
                        prepare_environment()
                    _preflight_environment(params, environment_kwargs=environment_kwargs)
            provider_name, model_name = safe_llm_labels(provider.provider, provider.model)
            echo(f"Autopilot: generating one case with {provider_name} / {model_name}.")

        dataset_note = "auto-generated by autopilot — review evals/"
        generated = False
        with dataset_generation_progress(
            enabled=not quiet and progress != "off", stream=click.get_text_stream("stderr")
        ):
            try:
                if no_llm:
                    service.create_autopilot_dataset(
                        skill_path,
                        use_llm=False,
                        fallback_reason="public LLM provider is not configured",
                    )
                else:
                    service.create_autopilot_dataset(skill_path, use_llm=True)
                generated = True
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
                    service.create_autopilot_dataset(skill_path, use_llm=False, fallback_reason=diagnostic)
                    generated = True
            if not eval_source_exists():
                raise click.ClickException("Autopilot dataset generation did not produce an evaluation source.")
        if generated:
            guidance = starter_guidance(skill_path)
            if guidance is not None:
                echo(strip_terminal_controls(guidance))
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


def _catalog_results_dir(ctx: click.Context, skill_dir: Path, catalog_root: Path) -> Path | None:
    """Preserve catalog-relative parents under an external Tier 3 results root."""
    import os

    from skillevaluator.tier3.results_location import ENV_RESULTS_DIR

    configured = ctx.params.get("results_dir")
    if configured is None:
        configured = os.environ.get(ENV_RESULTS_DIR)
    if not configured:
        return None
    return Path(configured).expanduser() / skill_dir.relative_to(catalog_root).parent


def _catalog_child_argv_from_ctx(
    ctx: click.Context, skill_dir: Path, output_dir: Path, *, catalog_root: Path | None = None
) -> list[str]:
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
    if params.get("skip"):
        argv.extend(["--skip", str(params["skip"])])
    if params.get("previous_version"):
        argv.extend(["--previous-version", str(params["previous_version"])])
    if params.get("harness"):
        argv.extend(["--harness", str(params["harness"])])
    if params.get("harness_profiles"):
        argv.extend(["--harness-profiles", str(params["harness_profiles"])])
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
    if params.get("skillspector_baseline"):
        argv.extend(["--skillspector-baseline", str(params["skillspector_baseline"])])
    if not params.get("dedup", True):
        argv.append("--no-dedup")
    block_on_dedup = params.get("block_on_dedup")
    if block_on_dedup is True:
        argv.append("--block-on-dedup")
    elif block_on_dedup is False:
        argv.append("--no-block-on-dedup")
    min_score = params.get("min_score", QUALITY_DEFAULT_MIN_SCORE)
    if min_score != QUALITY_DEFAULT_MIN_SCORE:
        argv.extend(["--min-score", str(min_score)])
    if params.get("quality_config"):
        argv.extend(["--quality-config", str(Path(params["quality_config"]).resolve())])
    if params.get("pii_waivers"):
        # Not resolved: a waiver file's entries are relative to the directory the file (or the
        # symlink to it) is in, so a child must see the same path the parent was given.
        argv.extend(["--pii-waivers", str(Path(params["pii_waivers"]).absolute())])
    if params.get("external"):
        argv.append("--external")
    if params.get("policy_path"):
        argv.extend(["--policy", str(params["policy_path"])])
    if params.get("profile"):
        argv.extend(["--profile", str(params["profile"])])
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
    if params.get("skip_baseline"):
        argv.append("--skip-baseline")
    if params.get("n_concurrent") is not None:
        argv.extend(["--n-concurrent", str(params["n_concurrent"])])
    if params.get("max_agents") is not None:
        argv.extend(["--max-agents", str(params["max_agents"])])
    if params.get("max_cases") is not None:
        argv.extend(["--max-cases", str(params["max_cases"])])
    if params.get("n_attempts") is not None:
        argv.extend(["--n-attempts", str(params["n_attempts"])])
    if params.get("max_agent_tokens") is not None:
        argv.extend(["--max-agent-tokens", str(params["max_agent_tokens"])])
    if params.get("max_judge_calls") is not None:
        argv.extend(["--max-judge-calls", str(params["max_judge_calls"])])
    if params.get("max_attempts_total") is not None:
        argv.extend(["--max-attempts-total", str(params["max_attempts_total"])])
    if params.get("max_cost_usd") is not None:
        argv.extend(["--max-cost-usd", str(params["max_cost_usd"])])
    if params.get("price_file") is not None:
        argv.extend(["--price-file", str(Path(params["price_file"]).resolve())])
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
    results_dir = _catalog_results_dir(ctx, skill_dir, catalog_root) if catalog_root else params.get("results_dir")
    if results_dir is not None:
        argv.extend(["--results-dir", str(results_dir)])
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
    tolerate_baseline_timeouts = params.get("tolerate_baseline_timeouts")
    if tolerate_baseline_timeouts is True:
        argv.append("--tolerate-baseline-timeouts")
    elif tolerate_baseline_timeouts is False:
        argv.append("--no-tolerate-baseline-timeouts")
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
    if catalog_root is not None:
        argv.extend(["--catalog-root", str(catalog_root)])
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
    entry["report_scope"] = "single_skill"
    try:
        payload = json.loads(json_report.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return entry
    if not isinstance(payload, dict):
        return entry

    for key in ("overall_passed", "overall_status", "incomplete_scans", "severity_counts"):
        if key in payload:
            entry[key] = payload[key]
    entry["report_verdict"] = {key: payload[key] for key in ("overall_passed", "overall_status") if key in payload}
    return entry


def _catalog_entry_with_metadata_id_conflicts(
    skill: dict[str, object], conflicts: list[dict[str, object]]
) -> dict[str, object]:
    """Project catalog findings onto a member, retaining its independent report verdict.

    The findings themselves are listed under ``catalog_findings`` so the
    verdict and the severity bump above can be traced back to what caused
    them: a catalog validates each skill on its own, so these findings exist
    nowhere else in that skill's output.
    """
    entry = dict(skill)
    recorded: list[dict[str, object]] = []
    for conflict in conflicts:
        members = conflict.get("skills")
        if not isinstance(members, list) or entry.get("name") not in members:
            continue
        names = [str(member) for member in members]
        others = ", ".join(name for name in names if name != entry.get("name"))
        recorded.append(
            {
                "category": "SCHEMA",
                "check_name": "metadata_id_duplicate",
                "severity": conflict.get("severity"),
                "blocking": bool(conflict.get("blocking")),
                "message": f"metadata.id '{conflict.get('metadata_id')}' is also declared by {others}",
            }
        )
        severity = conflict.get("severity")
        if severity in ("critical", "high", "medium", "low"):
            raw_counts = entry.get("severity_counts")
            counts = dict(raw_counts) if isinstance(raw_counts, dict) else {}
            previous = counts.get(severity, 0)
            counts[severity] = (previous if isinstance(previous, int) else 0) + 1
            entry["severity_counts"] = counts
        if conflict.get("blocking"):
            entry["passed"] = False
            entry["overall_passed"] = False
            entry["overall_status"] = "failed"
    if recorded:
        entry["catalog_findings"] = recorded
    return entry


def _write_catalog_summary(
    output_dir: Path,
    skills: list[dict[str, object]],
    *,
    metadata_id_conflicts: list[dict[str, object]] | None = None,
) -> Path:
    """Write catalog-context verdicts and counts; linked leaf reports retain single-skill scope."""
    from skillevaluator.reporting.base import _write_report_atomically

    skills = [_catalog_entry_with_metadata_id_conflicts(skill, metadata_id_conflicts or []) for skill in skills]
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
        "metadata_id_conflicts": metadata_id_conflicts or [],
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }
    output_path = output_dir / CATALOG_SUMMARY_FILENAME
    payload = json.dumps(summary, indent=2, default=str, allow_nan=False).encode("utf-8")
    _write_report_atomically(output_path, payload)
    return output_path


def _catalog_metadata_id_conflicts(
    policy: ValidationPolicy | None,
    skill_dirs: list[Path],
    catalog_root: Path,
    discovery_problems: dict[Path, str],
    *,
    schema_enabled: bool = True,
) -> tuple[list[dict[str, object]], list[tuple[str, str]]]:
    """Find ``metadata.id`` values shared by catalog skills.

    Catalog runs validate each skill on its own, so uniqueness across the
    catalog is checked here. Returns the conflicts for the catalog summary and,
    when the policy makes a duplicate blocking, one failure per affected skill.

    This is part of the schema check, so it runs only when that check is
    selected: ``--checks`` and ``--skip`` run exactly the checks they name.
    """
    from skillevaluator.validators.policy import default_policy

    if not schema_enabled:
        return [], []
    from skillevaluator.validators.schema import (
        declared_metadata_id,
        duplicate_metadata_ids,
        metadata_id_duplicate_severity,
    )

    policy = policy or default_policy()
    declared: dict[str, list[Path]] = {}
    for skill_dir in skill_dirs:
        if skill_dir in discovery_problems:
            continue
        metadata_id = declared_metadata_id(skill_dir)
        if metadata_id is not None:
            declared.setdefault(metadata_id, []).append(skill_dir)
    duplicates = duplicate_metadata_ids(declared, catalog_root)
    if not duplicates:
        return [], []
    severity = metadata_id_duplicate_severity(policy)
    blocking = severity.is_error()
    conflicts: list[dict[str, object]] = [
        {"metadata_id": value, "skills": skills, "severity": severity.value, "blocking": blocking}
        for value, skills in duplicates.items()
    ]
    failures: list[tuple[str, str]] = []
    if blocking:
        for value, skills in duplicates.items():
            for skill in skills:
                others = ", ".join(other for other in skills if other != skill)
                failures.append((skill, f"metadata.id '{value}' is also declared by {others}"))
    return conflicts, failures


def _print_metadata_id_conflicts(conflicts: list[dict[str, object]]) -> None:
    """Show advisory (non-blocking) catalog ID conflicts under the scoreboard."""
    from rich.text import Text

    from skillevaluator.reporting.console_ui import MUTED, make_view_console

    console_ = make_view_console()
    for conflict in conflicts:
        skills = conflict.get("skills")
        if conflict.get("blocking") or not isinstance(skills, list):
            continue
        line = f"metadata.id '{conflict['metadata_id']}' is declared by {', '.join(map(str, skills))} (advisory)"
        console_.print(Text.assemble(("      warning     ", MUTED), (strip_terminal_controls(line), MUTED)))


def _record_metadata_id_failures(
    failures: list[tuple[str, str]], id_failures: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Add blocking ID conflicts to *failures*, keeping a skill's own failure reason first."""
    failed = {name for name, _reason in failures}
    return [*failures, *((name, reason) for name, reason in id_failures if name not in failed)]


def _validate_catalog(
    ctx: click.Context,
    *,
    resolved_target: Path,
    skill_dirs: list[Path],
    output_dir: Path,
    workers: int = 1,
    discovery_problems: dict[Path, str] | None = None,
    policy: ValidationPolicy | None = None,
) -> None:
    """Run the full validate pipeline once per skill in the catalog.

    Each skill is an independent job with its own pipeline view, reports
    (under ``<output_dir>/<skill>/``), and verdict; the catalog exits nonzero
    when any skill failed. With ``workers`` above 1, skills validate in
    parallel child processes and the per-skill pipeline view is skipped.
    """
    if ctx.params.get("previous_version"):
        raise click.ClickException(
            "--previous-version applies to one skill and cannot be reused for a catalog; "
            "validate each skill separately with its own previous version"
        )

    discovery_problems = discovery_problems or {}
    if workers > 1:
        _validate_catalog_parallel(
            ctx,
            skill_dirs=skill_dirs,
            catalog_root=resolved_target,
            output_dir=output_dir,
            workers=workers,
            discovery_problems=discovery_problems,
            policy=policy,
        )
        return

    failures = [(path.relative_to(resolved_target).as_posix(), reason) for path, reason in discovery_problems.items()]
    meta_key = "skillevaluator_catalog_report_root"
    previous_catalog_root = ctx.meta.get(meta_key)
    ctx.meta[meta_key] = output_dir
    skill_reports: dict[str, str | None] = {}
    try:
        for index, skill_dir in enumerate(skill_dirs, start=1):
            skill_name = skill_dir.relative_to(resolved_target).as_posix()
            _print_catalog_divider(index, len(skill_dirs), skill_name)
            if skill_dir in discovery_problems:
                continue
            skill_output = output_dir / skill_name
            overrides = {
                **ctx.params,
                "target_path": skill_dir,
                "content_type": "skill",
                "output_dir": skill_output,
                "results_dir": _catalog_results_dir(ctx, skill_dir, resolved_target),
                "catalog_root": resolved_target,
            }
            reason = ""
            try:
                ctx.invoke(validate, **overrides)
            except click.ClickException as exc:
                reason = str(getattr(exc, "message", exc))
                failures.append((skill_name, reason))
            except Exception as exc:  # unexpected: keep the catalog running, report it on the scoreboard
                reason = f"unexpected error: {exc}"
                failures.append((skill_name, reason))
            skill_reports[skill_name] = _consume_validate_json_report()
    finally:
        if previous_catalog_root is None:
            ctx.meta.pop(meta_key, None)
        else:
            ctx.meta[meta_key] = previous_catalog_root
    id_conflicts, id_failures = _catalog_metadata_id_conflicts(
        policy,
        skill_dirs,
        resolved_target,
        discovery_problems,
        schema_enabled=runs_schema_check(ctx.params.get("checks"), ctx.params.get("skip")),
    )
    failures = _record_metadata_id_failures(failures, id_failures)
    failure_map = dict(failures)
    skill_entries = [
        _catalog_skill_entry(
            skill_name,
            output_dir / skill_name,
            passed=skill_name not in failure_map,
            reason=failure_map.get(skill_name, ""),
            json_report_name=skill_reports.get(skill_name),
        )
        for skill_name in (skill_dir.relative_to(resolved_target).as_posix() for skill_dir in skill_dirs)
    ]
    _write_catalog_summary(output_dir, skill_entries, metadata_id_conflicts=id_conflicts)
    _print_catalog_summary(len(skill_dirs), failures, output_dir)
    _print_metadata_id_conflicts(id_conflicts)
    if failures:
        raise click.ClickException(
            f"{len(failures)}/{len(skill_dirs)} skills failed validation: "
            + ", ".join(strip_terminal_controls(name) for name, _reason in failures)
        )


def _validate_catalog_parallel(
    ctx: click.Context,
    *,
    skill_dirs: list[Path],
    catalog_root: Path,
    output_dir: Path,
    workers: int,
    discovery_problems: dict[Path, str] | None = None,
    policy: ValidationPolicy | None = None,
) -> None:
    """Validate catalog skills concurrently in isolated child processes."""
    from skillevaluator.reporting.console_ui import make_view_console

    discovery_problems = discovery_problems or {}
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
            "skill_name": skill_name,
            "argv": _catalog_child_argv_from_ctx(ctx, skill_dir, output_dir / skill_name, catalog_root=catalog_root),
            "cwd": workdir,
            "output_dir": str(output_dir / skill_name),
        }
        for skill_dir in skill_dirs
        if skill_dir not in discovery_problems
        for skill_name in [skill_dir.relative_to(catalog_root).as_posix()]
    ]
    failures = [(path.relative_to(catalog_root).as_posix(), reason) for path, reason in discovery_problems.items()]
    worker_reports: dict[str, str | None] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_catalog_skill_worker, job) for job in jobs]
        for future in as_completed(futures):
            skill_name, passed, reason, json_report_name = future.result()
            worker_reports[skill_name] = json_report_name
            if not passed:
                failures.append((skill_name, reason))

    id_conflicts, id_failures = _catalog_metadata_id_conflicts(
        policy,
        skill_dirs,
        catalog_root,
        discovery_problems,
        schema_enabled=runs_schema_check(ctx.params.get("checks"), ctx.params.get("skip")),
    )
    failures = _record_metadata_id_failures(failures, id_failures)
    failures.sort(key=lambda item: item[0])
    failure_map = dict(failures)
    skill_entries = [
        _catalog_skill_entry(
            skill_name,
            output_dir / skill_name,
            passed=skill_name not in failure_map,
            reason=failure_map.get(skill_name, ""),
            json_report_name=worker_reports.get(skill_name),
        )
        for skill_name in (skill_dir.relative_to(catalog_root).as_posix() for skill_dir in skill_dirs)
    ]
    _write_catalog_summary(output_dir, skill_entries, metadata_id_conflicts=id_conflicts)
    _print_catalog_summary(len(skill_dirs), failures, output_dir)
    _print_metadata_id_conflicts(id_conflicts)
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
    presentation_context: FinalRenderContext | None = None,
) -> None:
    """Render the quiet-mode verdict panel and report footer."""
    from skillevaluator.reporting.console_ui import MUTED, Verdict, _is_skipped, first_fix
    from skillevaluator.reporting.harbor_viewer import harbor_job_links
    from skillevaluator.reporting.presentation import build_final_render_context
    from skillevaluator.reporting.run_status import Status

    context = presentation_context or build_final_render_context(
        [
            *tier_gate_results,
            *([tier3_result] if tier3_result is not None and tier3_result not in tier_gate_results else []),
        ]
    )
    details = tuple(f"Tier {number}: {tier.summary}" for number, tier in context.tiers.items() if tier.displayed)
    if context.tiers[3].status != Status.NOT_RUN:
        details += (context.coverage.text, *context.coverage.details)
    details += (f"Publication eligibility: {context.publication_label}",)

    ext = {"html": ".html", "json": ".json", "markdown": ".md", "sarif": ".sarif.json"}
    links: list[tuple[str, str]] = [
        ("report" if fmt == "html" else fmt, str(output_dir / f"{basename}{ext[fmt]}"))
        for fmt in report_formats
        if fmt in ext
    ]
    payload = ((tier3_result.metadata or {}).get("agent_eval") or {}) if tier3_result is not None else {}
    summary = payload.get("summary") or payload
    for job in harbor_job_links(payload):
        for key, label in (("url", "Results"), ("analysis_url", "Analysis")):
            if job.get(key):
                links.append((f"{job['name']} {label}", job[key]))

    if not gate_failed:
        advisory_failed = any(block.status == "fail" and block.number in {2, 3} for block in view.blocks)
        if advisory_failed:
            details += ("advisory tier reported findings (see report)",)
        lift = summary.get("overall_lift")
        if agent_eval and isinstance(lift, int | float) and not isinstance(lift, bool) and math.isfinite(lift):
            details += (f"skill lift {lift:+.2f}",)
        view.finish(Verdict(passed=True, headline=context.command_label, details=details), links)
        return

    failed = [result for result in tier_gate_results if not result.passed and not _is_skipped(result)]
    if len(failed) == 1:
        only = failed[0]
        tier_no = (only.metadata.get("gating") or {}).get("tier")
        state = "incomplete" if context.result_statuses[id(only)] == "incomplete" else "failed"
        headline = f"{only.validator_name or 'validation'} {state}" + (
            f" in Tier {tier_no}" if tier_no is not None else ""
        )
    else:
        by_tier: dict[object, list] = {}
        for result in failed:
            by_tier.setdefault((result.metadata.get("gating") or {}).get("tier"), []).append(result)
        parts = []
        for tier_no, results in by_tier.items():
            broken = sum(1 for r in results if context.result_statuses[id(r)] != "incomplete")
            counts = [
                f"{broken} failed" if broken else "",
                f"{len(results) - broken} incomplete" if broken < len(results) else "",
            ]
            parts.append((f"Tier {tier_no}: " if tier_no is not None else "") + ", ".join(c for c in counts if c))
        headline = " · ".join(parts) or "validation failed"
    rerun = _rerun_hint(target_path, agent_eval)
    fix = [("see what to fix above, then rerun", MUTED)] if failed else first_fix(tier_gate_results)
    view.finish(
        Verdict(
            passed=False,
            headline=f"{context.command_label}: {headline}",
            fix=fix,
            rerun=rerun,
            incomplete=context.command_status == Status.INCOMPLETE,
            details=details,
        ),
        links,
    )


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
    "Choices: schema, version, security, pii, license, code-integrity, unicode, quality, lint, harness; "
    "opt-in (not run by default): dependency. "
    "quality/lint/version are skill-only and skipped for rules/workflows.",
)
@click.option(
    "--harness",
    "harness",
    default=None,
    cls=GroupedOption,
    help_group=_RUN_GROUP,
    help="Harnesses for the advisory harness check (default: all), e.g. --harness codex,claude-code.",
)
@click.option(
    "--harness-profiles",
    "harness_profiles",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    cls=GroupedOption,
    help_group=_RUN_GROUP,
    help="Directory of harness profile YAMLs to use instead of the packaged ones.",
)
@click.option(
    "--skip",
    default=None,
    metavar="CHECKS",
    callback=_validate_skip,
    cls=GroupedOption,
    help_group=_TIER1_GROUP,
    help="Comma-separated Tier 1 checks to leave out, e.g. secrets (the Gitleaks scan). "
    "Skipped checks are shown and recorded as skipped in the report, and nothing is installed for them.",
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
    "--skillspector-baseline",
    type=SKILLSPECTOR_BASELINE_PATH,
    default=None,
    cls=GroupedOption,
    help_group=_TIER1_GROUP,
    help=_SKILLSPECTOR_BASELINE_HELP,
)
@click.option(
    "--min-score",
    type=int,
    default=QUALITY_DEFAULT_MIN_SCORE,
    show_default=True,
    cls=GroupedOption,
    help_group=_TIER1_GROUP,
    help="Minimum quality score (0-100) required to pass when the 'quality' check runs.",
)
@click.option(
    "--quality-config",
    type=QUALITY_CONFIG_PATH,
    default=None,
    cls=GroupedOption,
    help_group=_TIER1_GROUP,
    help="Override the packaged quality-default.yaml with a custom YAML file.",
)
@click.option(
    "--pii-waivers",
    type=PII_WAIVERS_PATH,
    default=None,
    cls=GroupedOption,
    help_group=_TIER1_GROUP,
    help=_PII_WAIVERS_HELP,
)
@click.option(
    "--profile",
    default=None,
    cls=GroupedOption,
    help_group=_RUN_GROUP,
    help=(
        "Validation policy: default or registry (external is a compatibility alias for default). "
        "Default: $SKILLEVALUATOR_PROFILE, then default."
    ),
)
@click.option(
    "--external",
    is_flag=True,
    cls=GroupedOption,
    help_group=_RUN_GROUP,
    help="Compatibility alias for --profile default; does not change repository visibility.",
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
# Set by a catalog run on each per-skill job: the folder of skills that the job's
# skill belongs to, which the security check judges missing references against.
@click.option("--catalog-root", type=click.Path(file_okay=False, path_type=Path), default=None, hidden=True)
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
    help="Make Tier 3 findings gate the exit code. Default: advisory.",
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
        "Comma-separated Harbor agents (hermes is experimental, Docker/NVIDIA Build only). Default follows the provider: "
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
    "--max-cases",
    type=click.IntRange(min=1),
    default=None,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Run only the first N eval cases (default: every case); a partial run is not publication evidence.",
)
@click.option(
    "--n-attempts",
    type=int,
    default=None,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Attempts per eval case (pass@k).",
)
@click.option("--max-agent-tokens", type=click.IntRange(min=0), default=None, help="Maximum observed agent tokens.")
@click.option(
    "--max-judge-calls",
    type=click.IntRange(min=0),
    default=None,
    help=(
        "Maximum recorded non-salvage judge attempts; transport retries and RAGAS-internal requests are not counted "
        "separately."
    ),
)
@click.option(
    "--max-attempts-total",
    type=click.IntRange(min=0),
    default=None,
    help="Maximum logical Harbor attempts across agents and arms, including private comparison arms.",
)
@click.option(
    "--max-cost-usd",
    type=click.FloatRange(min=0),
    default=None,
    help="Maximum agent model spend; requires --price-file and excludes judge spend, setup, and container time.",
)
@click.option(
    "--price-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Model price JSON with input_per_mtok/output_per_mtok rates and optional cached_input_per_mtok.",
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
    "--tolerate-baseline-timeouts/--no-tolerate-baseline-timeouts",
    default=None,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help=(
        "Keep an agent scored on its clean with-skill arm when only baseline attempts hit the agent "
        "execution timeout; its baseline and uplift are not reported [default: disabled]."
    ),
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
@_plugin_static_options
@_plugin_run_options
@_report_options
def validate(
    target_path: Path,
    content_type: str,
    tiers: str | None,
    full: bool,
    verbose: bool,
    checks: str | None,
    skip: str | None,
    previous_version: str | None,
    fail_fast: bool,
    continue_on_failure: bool,
    llm: bool,
    llm_verify: bool,
    skillspector_baseline: Path | None,
    min_score: int,
    quality_config: Path | None,
    pii_waivers: Path | None,
    profile: str | None,
    external: bool,
    policy_path: Path | None,
    dedup: bool,
    block_on_dedup: bool | None,
    agent_eval: bool | None,
    block_on_agent_eval: bool | None,
    autopilot: bool | None,
    agents: str | None,
    env_mode: str,
    environment_kwarg: tuple[str, ...],
    skip_baseline: bool,
    n_concurrent: int | None,
    max_agents: int | None,
    max_cases: int | None,
    n_attempts: int | None,
    max_agent_tokens: int | None,
    max_judge_calls: int | None,
    max_attempts_total: int | None,
    max_cost_usd: float | None,
    price_file: Path | None,
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
    tolerate_baseline_timeouts: bool | None,
    workers: int,
    catalog_root: Path | None,
    evaluated_source_repository: str | None,
    evaluated_source_revision: str | None,
    evaluator_container_revision: str | None,
    repo_root: Path | None,
    resolve_endpoints: bool,
    claude_validate: bool,
    plugin_load: str | None,
    lift_mode: str | None,
    probe_mcp: bool | None,
    probe_mcp_env: tuple[str, ...],
    trial_retries: int | None,
    report_formats: tuple[str, ...],
    output_dir: Path,
    harness: str | None = None,
    harness_profiles: Path | None = None,
) -> None:
    """Validate a skill through all three tiers, with automatic dataset preparation.

    Runs Tier 1 static, security, and quality checks (which gate the exit code),
    blocking Tier 2 deduplication, and advisory Tier 3 live evaluation.
    For skills, reuse the existing dataset or generate one when missing.
    Use --tiers 1,2 to omit live evaluation, or --block-on-agent-eval to make
    Tier 3 gate the exit code. Reports follow --report and --output-dir.

    Rules, workflows, and plugins retain Tier 1/2 by default; their Tier 3
    evaluation requires an explicit --tier3, --autopilot, or --full request.

    Plugins are auto-detected from the supported client manifests. Whole-tree
    checks and applicable bundled-skill quality/lint/version checks are included.
    """
    _record_validate_json_report(None)
    if harness:
        from skillevaluator.harness import load_profiles

        try:
            load_profiles(harness, harness_profiles)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="--harness") from exc
    if dedup:
        _reject_linked_tier2_root(target_path)
    from skillevaluator.cli_core import detect_content_type

    resolved_type = content_type
    if content_type == "auto":
        # Folder-name detection (``skills/``, ``team-rules/``...) needs an absolute path, so detect
        # on the resolved target as before. The raw argument only upgrades the result to a plugin:
        # resolving a symlinked plugin root or manifest would hide the link the plugin checks refuse.
        resolved_type = detect_content_type(target_path.resolve())
        if resolved_type != "plugin" and detect_content_type(target_path) == "plugin":
            resolved_type = "plugin"
    target_path = target_path.absolute() if resolved_type == "plugin" else target_path.resolve()
    from skillevaluator.publication_identity import publication_target_name

    # A case- or normalization-insensitive filesystem lets an alias such as
    # ``SIMPLE`` open the entry recorded as ``simple``. Every tier stamps the
    # recorded spelling, so reports must expect it instead of the alias.
    publication_name = publication_target_name(target_path) or target_path.name
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

    if external and profile and profile not in {"default", "external"}:
        raise click.ClickException(f"--external conflicts with --profile {profile}; pass one or the other.")
    profile_name = "external" if external else profile
    try:
        policy = resolve_policy(profile=profile_name, policy_path=policy_path)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    block_on_dedup_effective = True if block_on_dedup is None else block_on_dedup
    block_on_agent_eval_effective = False if block_on_agent_eval is None else block_on_agent_eval

    plugin_options = None
    plugin_source_path = target_path
    if resolved_type == CONTENT_TYPE_PLUGIN:
        from skillevaluator.cli_core import resolve_plugin_path
        from skillevaluator.evaluation.options import PluginOptions

        target_path = resolve_plugin_path(target_path)
        publication_name = publication_target_name(target_path) or target_path.name
        plugin_options = PluginOptions(
            plugin_load=plugin_load,
            lift_mode=lift_mode,
            probe_mcp=probe_mcp,
            probe_mcp_env=probe_mcp_env,
            trial_retries=trial_retries,
            repo_root=repo_root,
            allowed_private_hosts=policy.mcp_allowed_private_hosts,
            hook_allowed_urls=policy.hook_allowed_urls,
        )
    elif (
        any(value is not None for value in (plugin_load, lift_mode, probe_mcp, trial_retries, repo_root))
        or probe_mcp_env
        or resolve_endpoints
        or claude_validate
    ):
        raise click.UsageError("Plugin controls require a plugin target (--type plugin).")
    # Only skills own an evals/ task source. Plugins, rules, and workflows can
    # still run Tier 3, but must bypass the skill-directory source preflight.
    preflight_tier3_source = resolved_type == CONTENT_TYPE_SKILL

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
    autopilot = autopilot and agent_eval and resolved_type != CONTENT_TYPE_PLUGIN
    tier3_path = (
        plugin_source_path
        if resolved_type == CONTENT_TYPE_PLUGIN
        else (target_path.parent if preflight_tier3_source and target_path.is_file() else target_path)
    )

    from skillevaluator.constants import CONTENT_TYPE_UNKNOWN, SKILL_MANIFEST_VARIANTS
    from skillevaluator.utils.helpers import find_skills_in_directory

    catalog_skill_dirs: list[Path] = []
    discovery_problems: dict[Path, str] = {}
    if (
        resolved_type in (CONTENT_TYPE_SKILL, CONTENT_TYPE_UNKNOWN)
        and target_path.is_dir()
        and not any(
            (target_path / name).exists() or (target_path / name).is_symlink() for name in SKILL_MANIFEST_VARIANTS
        )
    ):
        catalog_skill_dirs = find_skills_in_directory(target_path, problems=discovery_problems)
    is_catalog = bool(catalog_skill_dirs)
    # Quiet (default) swaps the implicit CLI reporter for HTML+JSON, while an
    # explicit -r cli is a no-file contract. Skills and catalogs additionally
    # require storage for their compulsory BENCHMARK.md output.
    quiet = not verbose and not logging.getLogger().isEnabledFor(logging.DEBUG)
    file_report_required = any(report_format in _FILE_REPORT_EXTENSIONS for report_format in report_formats) or (
        quiet and not _report_formats_explicit()
    )
    report_output_required = (
        resolved_type in {CONTENT_TYPE_SKILL, CONTENT_TYPE_PLUGIN} or is_catalog or file_report_required
    )
    if report_output_required:
        output_dir = _resolve_report_output_location(target_path, output_dir)
        _reject_copy_repo_root_output(
            target_path,
            output_dir,
            copy_repo=copy_repo,
            agent_eval=agent_eval,
            include_skills=include_skills,
        )

    # A directory of skills (no root SKILL.md) is a catalog: run the pipeline
    # once per skill, each as its own job with its own reports.
    if is_catalog:
        _validate_catalog(
            click.get_current_context(),
            resolved_target=target_path,
            skill_dirs=catalog_skill_dirs,
            output_dir=output_dir,
            workers=workers,
            discovery_problems=discovery_problems,
            policy=policy,
        )
        return

    # Quiet (default) drives the compact pipeline view; --verbose keeps the
    # historical full-detail stream, as does DEBUG logging via the group -v.
    run_tier3 = agent_eval
    planned_tiers = [(1, "Static & Security", "static & security")]
    tier2_index = tier3_index = None
    if dedup:
        tier2_index = len(planned_tiers)
        planned_tiers.append((2, "Deduplication", "deduplication"))
    if run_tier3:
        tier3_index = len(planned_tiers)
        planned_tiers.append((3, "Live Agent Eval", "live agent eval"))
    view = ValidateView(
        skill=f"{resolved_type}: {target_path.name}",
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

    if run_tier3 and env_mode == "local":
        _offer_local_runtime_install(agents)

    # Autopilot source generation must precede every tier's publication-target
    # capture so one combined report cannot bind otherwise valid tier evidence
    # to different source trees. Generation remains advisory: retain any error
    # for the later Tier 3 skip result and continue through Tier 1/2.
    autopilot_error: str | None = None
    autopilot_dataset_note: str | None = None
    _offer_scanner_install(checks, skip)
    view.start()
    if autopilot:
        # Generation now precedes Tier 1, so surface it on the first live block
        # instead of leaving the pipeline view blank while the dataset is prepared.
        view.tier_start(0)
        view.tier_progress(0, [stage_hint_row("status", "preparing evaluation dataset…")])
        try:
            autopilot_dataset_note = _ensure_autopilot_dataset(
                tier3_path, quiet=quiet, evaluation_params=click.get_current_context().params
            )
        except (Exception, SystemExit) as exc:
            try:
                error_detail = sanitize_tier3_console_text(str(exc), limit=1024)
            except Exception:
                error_detail = type(exc).__name__
            autopilot_error = sanitize_tier3_console_text(
                f"autopilot dataset generation failed: {error_detail or type(exc).__name__}", limit=1200
            )
            if not quiet:
                click.echo(f"Warning: {autopilot_error}", err=True)

    view.tier_start(0)
    check_lineup = enabled_check_lineup(checks)
    checks_done: list[str] = []

    def _on_check(name: str) -> None:
        view.tier_progress(0, [check_ticker_row(check_lineup, checks_done, name)])
        checks_done.append(name)

    results = run_validation(
        plugin_source_path if resolved_type == CONTENT_TYPE_PLUGIN else target_path,
        checks=checks,
        use_llm=llm,
        llm_verify=llm_verify,
        skillspector_baseline=skillspector_baseline,
        min_score=min_score,
        quality_config=quality_config,
        pii_waivers=pii_waivers,
        previous_version=previous_version,
        policy=policy,
        content_type=resolved_type,
        target_root=catalog_root.resolve() if catalog_root is not None else None,
        fail_fast=fail_fast,
        continue_on_failure=continue_on_failure,
        on_check=_on_check if quiet else None,
        harness=harness,
        harness_profiles=harness_profiles,
        skip=skip,
        **(
            {"repo_root": repo_root, "resolve_endpoints": resolve_endpoints, "claude_validate": claude_validate}
            if resolved_type == CONTENT_TYPE_PLUGIN
            else {}
        ),
    )
    # The raw pass/fail signal drives --fail-fast identically in both modes;
    # the DISPLAYED tier summary must reflect policy-finalized severities or
    # the tier blocks can contradict the verdict panel (apply_policy is
    # idempotent, so emit_reports re-applying it later is a no-op).
    tier1_raw_failed = any(not r.passed for r in results)
    if quiet:
        apply_policy(results, policy)
    tier1_ok, tier1_rows = summarize_tier1(results, lineup=check_lineup)
    harness_rows = harness_block(results)
    if harness_rows:
        view.set_harness(harness_panel(harness_rows, width=view.width, verbose=verbose), harness_note(harness_rows))
    view.tier_done(0, failed=not tier1_ok, rows=tier1_rows)
    tier1_gate_results = list(results)
    tier2_gate_results: list[ValidationResult] = []

    if dedup and not (fail_fast and not continue_on_failure and tier1_raw_failed):
        if not quiet:
            _print_tier_banner(_TIER_BANNERS["tier2"])
        view.tier_start(tier2_index)
        view.tier_progress(tier2_index, [stage_hint_row("stages", "chunk · embed · cluster · llm-judge")])
        tier2_results = _run_dedup_or_skip(plugin_source_path if resolved_type == CONTENT_TYPE_PLUGIN else target_path)
        results.extend(tier2_results)
        tier2_gate_results.extend(tier2_results)
        if quiet:
            apply_policy(tier2_results, policy)
        tier2_ran, tier2_ok, tier2_rows, tier2_skip = summarize_tier2(tier2_results)
        if tier2_ran:
            view.tier_done(tier2_index, failed=not tier2_ok, rows=tier2_rows)
        else:
            view.tier_skip(tier2_index, tier2_skip)
    elif dedup:
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
    if not quiet and agent_eval and "cli" in report_formats:
        apply_policy(tier_gate_results, policy)
        CLIReporter(console=console).print_summary(tier_gate_results)

    # Tier 3 runs BEFORE report emission so its results are folded into the
    # single combined HTML/JSON/BENCHMARK.md report (parity with SkillEvaluator), and
    # runs regardless of Tier 1/Tier 2 outcome. It degrades to a non-blocking
    # advisory note when it cannot run.
    tier3_result: ValidationResult | None = None
    if agent_eval:
        if not quiet:
            _print_tier_banner(_TIER_BANNERS["tier3"])
        view.tier_start(tier3_index)
        env_note = {
            "docker": "isolated containers per trial",
            "kata": "isolated Kata microVMs per trial",
            "local": "experimental host sandbox — trusted skills and workspaces only",
        }.get(env_mode, "")
        model_display = ", ".join(agent_model) if agent_model else (model or "agent defaults")
        tier3_config_rows = [
            detail_row("agent", agents or "provider-native default"),
            detail_row("env", env_mode, env_note),
            detail_row("model", model_display),
        ]

        if autopilot_dataset_note:
            tier3_config_rows.append(detail_row("dataset", autopilot_dataset_note))

        view.tier_progress(
            tier3_index,
            [*tier3_config_rows, stage_hint_row("status", "running with-skill and baseline trials…")],
        )

        from threading import Lock

        live_case_rows: list[TierRow] = []
        engine_tail: list[str] = []
        progress_lock = Lock()  # case rows refresh from a polling thread, engine lines from the runner

        def _show_tier3_progress() -> None:
            with progress_lock:
                view.tier_progress(tier3_index, [*tier3_config_rows, *live_case_rows, *engine_feed_rows(engine_tail)])

        def _on_engine_tail(lines: list[str]) -> None:
            engine_tail[:] = lines
            _show_tier3_progress()

        def _on_case_rows(rows: list[TierRow]) -> None:
            live_case_rows[:] = rows
            _show_tier3_progress()

        reporter = (
            ViewProgressReporter(
                _on_engine_tail,
                on_cases=_on_case_rows,
                pass_threshold=pass_threshold,
                progress_reporter=view.progress_reporter,
            )
            if quiet
            else None
        )
        repo_context_exclude_paths = [output_dir] if report_output_required else []
        catalog_report_root = click.get_current_context().meta.get("skillevaluator_catalog_report_root")
        if isinstance(catalog_report_root, Path) and not paths_refer_to_same_location(
            catalog_report_root,
            output_dir,
        ):
            repo_context_exclude_paths.append(catalog_report_root)
        tier3_result = _run_agent_eval_or_skip(
            tier3_path,
            agents=agents,
            env_mode=env_mode,
            environment_kwarg=environment_kwarg,
            skip_baseline=skip_baseline,
            n_concurrent=n_concurrent,
            max_agents=max_agents,
            max_cases=max_cases,
            n_attempts=n_attempts,
            max_agent_tokens=max_agent_tokens,
            max_judge_calls=max_judge_calls,
            max_attempts_total=max_attempts_total,
            max_cost_usd=max_cost_usd,
            price_file=price_file,
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
            tolerate_baseline_timeouts=tolerate_baseline_timeouts,
            block_on_agent_eval=block_on_agent_eval_effective,
            validate_source=preflight_tier3_source,
            evaluated_source=evaluated_source,
            progress_reporter=reporter,
            repo_context_exclude_paths=tuple(repo_context_exclude_paths),
            **({"plugin_options": plugin_options} if plugin_options is not None else {}),
        )
        results.append(tier3_result)
        if autopilot_error:
            # Invalid/missing sources may be reported as a failed Tier 3 check,
            # rather than a skip. Preserve the preparation diagnostic in both
            # cases so users can recover from the reason generation stopped.
            tier3_result.metadata["autopilot_error"] = autopilot_error
            tier3_result.add_warning(autopilot_error)
        # The panel follows the formats this run emits: with cli the full report
        # lists the source errors under the Tier 3 finding, so the panel keeps
        # the count; it shortens its own list only when an HTML or JSON report
        # keeps every error.
        tier3_ran, tier3_ok, tier3_rows, tier3_skip = summarize_tier3(
            tier3_result,
            report_formats=_effective_report_formats(report_formats, quiet=quiet),
        )
        if autopilot_error and not tier3_ran:
            from skillevaluator.reporting.base import get_skip_reason

            tier3_result.metadata["skip_reason"] = f"{autopilot_error}; {tier3_skip}"
            report_skip_reason = get_skip_reason(tier3_result)
            # Reports read the skip reason from metadata, so the generation
            # failure must land there too, not only in the view's skip row.
            tier3_result.metadata["skip_reason"] = report_skip_reason
            tier3_skip = sanitize_tier3_console_text(report_skip_reason)
        if tier3_ran:
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

            source_kind, _source_checks = validate_tier3_source(tier3_path)
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
            from skillevaluator.evaluation.tier3_report import fit_token_usage

            # Measured token usage makes room for these keys when the payload is at its limits.
            fit_token_usage(agent_eval_payload)
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
    target_display = resolve_git_remote_url(target_path) or str(target_path)
    sarif_repository_root = resolve_git_root(target_path)
    if sarif_repository_root is None:
        sarif_repository_root = target_path if target_path.is_dir() else target_path.parent

    # Quiet mode defaults the reports to html+json (the terminal shows only
    # the summary; the files carry the findings) and points at them from the
    # footer. An EXPLICIT -r is a contract and is honored exactly — including
    # "cli", which renders the full Rich report below the pipeline view.
    effective_formats = _effective_report_formats(report_formats, quiet=quiet)
    report_basename_value = make_timestamped_basename(f"{REPORT_PREFIX}-output")
    from skillevaluator.reporting.presentation import build_final_render_context

    presentation_context = build_final_render_context(
        results,
        requested_tiers=tuple(number for number, enabled in ((1, True), (2, dedup), (3, agent_eval)) if enabled),
        displayed_tiers=(1, 2, 3),
        expected_skill_name=publication_name,
    )
    emit_reports(
        results,
        presentation_context=presentation_context,
        report_formats=effective_formats,
        output_dir=output_dir,
        basename=report_basename_value,
        policy=policy,
        target_path=target_display,
        expected_skill_name=publication_name,
        # A Git remote URL in ``target_display`` cannot be probed for SKILL.md,
        # and a clean run has no finding path to infer the skill from.
        explicit_skill_name=publication_name if resolved_type == CONTENT_TYPE_SKILL and target_path.is_dir() else None,
        content_label=content_label,
        announce_paths=not quiet,
        sarif_scan_root=target_path,
        sarif_repository_root=sarif_repository_root,
    )
    _record_validate_json_report(f"{report_basename_value}.json" if "json" in effective_formats else None)
    if harness_rows and "json" in effective_formats:
        write_json_block(output_dir / f"{report_basename_value}.json", harness_rows)

    # BENCHMARK.md is generated compulsorily for skills (matches SkillEvaluator), even on
    # failure, so the publication card always reflects the latest evaluation --
    # now including Tier 3 results when --agent-eval ran.
    if resolved_type in {CONTENT_TYPE_SKILL, CONTENT_TYPE_PLUGIN}:
        from skillevaluator.reporting import BenchmarkReporter
        from skillevaluator.reporting.naming import BENCHMARK_FILENAME
        from skillevaluator.source_identity import EvaluatedSourceConflict

        output_dir.mkdir(parents=True, exist_ok=True)
        # ``_finalize_evaluated_source`` already attached and resolved the
        # identity, so a conflict aborts before any report file exists. This
        # stays as the last line of defence: the renderer re-validates the
        # carriers it is handed, and a producer can record one after the fact.
        try:
            BenchmarkReporter(skill_name=publication_name, evaluator_version=__version__).save(
                results, output_dir / BENCHMARK_FILENAME
            )
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
    from skillevaluator.reporting.run_status import Status, resolve_run_status

    gate_failed = resolve_run_status(effective_gate_results).headline != Status.PASSED
    if quiet:
        _finish_pipeline_view(
            view,
            tier_gate_results=effective_gate_results,
            tier3_result=tier3_result,
            gate_failed=gate_failed,
            output_dir=output_dir,
            basename=report_basename_value,
            report_formats=effective_formats,
            target_path=target_path,
            agent_eval=agent_eval,
            presentation_context=presentation_context,
        )
    if harness_rows and not quiet:
        console.print(harness_panel(harness_rows, verbose=True))
    if gate_failed:
        raise click.ClickException("validation failed")


# Intro text shown under the grouped Tier 3 options in ``validate --help``.
validate.help_group_descriptions = {
    _RUN_GROUP: _RUN_GROUP_DESC,
    _TIER1_GROUP: _TIER1_GROUP_DESC,
    _TIER2_GROUP: _TIER2_GROUP_DESC,
    _TIER3_GROUP: _TIER3_GROUP_DESC,
}


@cli.command("harness-check")
@_target_argument
@click.option(
    "--harness",
    default="all",
    show_default=True,
    help="Comma-separated harnesses to check, e.g. codex,claude-code.",
)
@click.option(
    "--harness-profiles",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="Directory of harness profile YAMLs to use instead of the packaged ones.",
)
@click.option("-v", "--verbose", is_flag=True, help="List every harness with its verified version and date.")
@click.option(
    "--json",
    "json_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Also write the harness_compatibility block to this JSON file.",
)
@click.option(
    "--context-tokens",
    type=click.IntRange(min=0),
    default=0,
    help="Context window of your model in tokens; some loader limits depend on it (0 = unknown, assume the smallest).",
)
@click.option(
    "--custom-provider",
    is_flag=True,
    help="Your model is served through a custom or OpenAI-compatible provider, or is a model the harness ships no metadata for.",
)
@click.option(
    "--explicit-invocation",
    is_flag=True,
    help="Skills are activated by an explicit mention (e.g. Codex $skill), not opened by the model.",
)
@click.option(
    "--headless",
    is_flag=True,
    help="The harness runs non-interactively (e.g. claude -p in CI) with default permissions.",
)
@click.option(
    "--harness-version",
    "harness_versions",
    multiple=True,
    metavar="HARNESS=VERSION",
    help="The release you run, e.g. codex=0.170.0; a release other than the profile's verified one is flagged.",
)
def harness_check(
    target_path: Path,
    harness: str,
    harness_profiles: Path | None,
    verbose: bool,
    json_path: Path | None,
    context_tokens: int,
    custom_provider: bool,
    explicit_invocation: bool,
    headless: bool,
    harness_versions: tuple[str, ...],
) -> None:
    """Show whether each agent harness loads, degrades or drops a skill (advisory; exits 0)."""
    from skillevaluator.harness.overlay import profile_names
    from skillevaluator.harness.report import run_harness_check
    from skillevaluator.reporting.console_ui import make_view_console

    try:
        config = {
            "context_tokens": context_tokens,
            "custom_provider": custom_provider,
            "explicit_invocation": explicit_invocation,
            "headless": headless,
            "harness_versions": dict(_split_harness_version(v) for v in harness_versions),
        }
        # A version may be given for any harness by its ``harness:`` name; a profile that won't parse can't fail this.
        known = set(profile_names(harness_profiles))
        if unknown := sorted(set(config["harness_versions"]) - known):
            raise click.BadParameter(
                f"unknown harness {', '.join(unknown)}; choose from {', '.join(sorted(known))}",
                param_hint="--harness-version",
            )
        block = run_harness_check(
            target_path, harness, harness_profiles, console=make_view_console(), verbose=verbose, config=config
        )
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--harness") from exc
    if not block:
        raise click.ClickException(f"No skills found in {target_path}")
    from skillevaluator.validators.schema import SchemaValidator

    for skill in block:
        schema = SchemaValidator().validate(Path(skill))
        if not schema.passed and schema.errors:
            click.echo(
                f"Tier 1 schema also fails for {Path(skill).name}: {schema.errors[0].split(' in /', 1)[0]}. "
                "Run `skillevaluator validate --tiers 1` for details."
            )
    if json_path is not None:
        json_path.write_text(json.dumps({"harness_compatibility": block}, indent=2), encoding="utf-8")


def _split_harness_version(value: str) -> tuple[str, str]:
    name, sep, version = value.partition("=")
    if not sep or not name.strip() or not version.strip():
        raise click.BadParameter(f"expected HARNESS=VERSION, got {value!r}", param_hint="--harness-version")
    return name.strip(), version.strip()


@cli.command("quality-check")
@_target_argument
@click.option("--min-score", type=int, default=QUALITY_DEFAULT_MIN_SCORE, show_default=True)
@click.option(
    "--quality-config",
    type=QUALITY_CONFIG_PATH,
    default=None,
    help="Override the packaged quality-default.yaml with a custom YAML file.",
)
@_report_options
def quality_check(
    target_path: Path,
    min_score: int,
    quality_config: Path | None,
    report_formats: tuple[str, ...],
    output_dir: Path,
) -> None:
    """Score skill quality across correctness, discoverability, reliability, and efficiency."""
    output_dir = _resolve_file_report_output_location(target_path.resolve(), output_dir, report_formats)
    if not emit_reports(
        run_quality_check(target_path, min_score=min_score, quality_config=quality_config),
        report_formats=report_formats,
        output_dir=output_dir,
        basename=report_basename("quality"),
    ):
        raise click.ClickException("quality check failed")


@cli.command("rubric-eval")
@_target_argument
@click.option("--min-score", type=int, default=QUALITY_DEFAULT_MIN_SCORE, show_default=True)
@_report_options
def rubric_eval(target_path: Path, min_score: int, report_formats: tuple[str, ...], output_dir: Path) -> None:
    """Run LLM-as-judge rubric evaluation for a skill."""
    output_dir = _resolve_file_report_output_location(target_path.resolve(), output_dir, report_formats)
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
@click.option(
    "--skillspector-baseline", type=SKILLSPECTOR_BASELINE_PATH, default=None, help=_SKILLSPECTOR_BASELINE_HELP
)
@_report_options
def security_scan(
    target_path: Path,
    llm: bool,
    llm_verify: bool,
    skillspector_baseline: Path | None,
    report_formats: tuple[str, ...],
    output_dir: Path,
) -> None:
    """Scan for security vulnerabilities."""
    output_dir = _resolve_file_report_output_location(target_path.resolve(), output_dir, report_formats)
    if not emit_reports(
        run_security_scan(target_path, use_llm=llm, llm_verify=llm_verify, skillspector_baseline=skillspector_baseline),
        report_formats=report_formats,
        output_dir=output_dir,
        basename=report_basename("security"),
    ):
        raise click.ClickException("security scan failed")


@cli.command("pii-scan")
@_target_argument
@click.option("--llm-verify", is_flag=True, help="Use LLM verification to reduce false positives.")
@click.option("--pii-waivers", type=PII_WAIVERS_PATH, default=None, help=_PII_WAIVERS_HELP)
@_report_options
def pii_scan(
    target_path: Path,
    llm_verify: bool,
    pii_waivers: Path | None,
    report_formats: tuple[str, ...],
    output_dir: Path,
) -> None:
    """Scan for PII and local identifiers."""
    output_dir = _resolve_file_report_output_location(target_path.resolve(), output_dir, report_formats)
    if not emit_reports(
        run_pii_scan(target_path, llm_verify=llm_verify, pii_waivers=pii_waivers),
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
    output_dir = _resolve_file_report_output_location(target_path.resolve(), output_dir, report_formats)
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
    "--type", "content_type", default="auto", type=click.Choice(["skill", "rules", "workflows", "plugin", "auto"])
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
    help="Compare one skill or plugin against a local catalog.",
)
@click.option(
    "--save-catalog",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Build and save a versioned local catalog from this collection.",
)
@click.option("--cache", type=click.Path(path_type=Path), default=None, hidden=True)
@click.option("--save-cache", type=click.Path(path_type=Path), default=None, hidden=True)
@click.option("--llm", "llm_verdict", is_flag=True, help="Request advisory LLM verdicts for plugin catalog matches.")
@click.option("--llm-model", default=None, help="Plugin catalog verdict model override.")
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
    llm_verdict: bool,
    llm_model: str | None,
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
    if llm_verdict or llm_model:
        from skillevaluator.tier2.plugin_commands import plugin_target

        if plugin_target(content_path, content_type) is None:
            raise click.UsageError("--llm and --llm-model require a plugin target (--type plugin).")

    _reject_linked_tier2_root(content_path)
    output_dir = _resolve_file_report_output_location(content_path.resolve(), output_dir, report_formats)
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
        **({"llm_verdict": llm_verdict, "llm_model": llm_model} if llm_verdict or llm_model else {}),
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
    output_dir = _resolve_file_report_output_location(skill_path.resolve(), output_dir, report_formats)
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
@click.option(
    "--catalog",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Compare plugin members against a local catalog.",
)
@click.option("--llm", "llm_verdict", is_flag=True, help="Request advisory LLM verdicts for plugin catalog matches.")
@_report_options
def dedup_scan(
    skill_path: Path,
    threshold: float,
    llm_model: str | None,
    model: str | None,
    catalog: Path | None,
    llm_verdict: bool,
    report_formats: tuple[str, ...],
    output_dir: Path,
) -> None:
    """Detect semantically redundant content within one skill."""
    from skillevaluator.tier2.commands import run_dedup_scan

    if catalog is not None or llm_verdict:
        from skillevaluator.tier2.plugin_commands import plugin_target

        if plugin_target(skill_path) is None:
            raise click.UsageError("--catalog and --llm require a plugin target.")
    _reject_linked_tier2_root(skill_path)
    output_dir = _resolve_file_report_output_location(skill_path.resolve(), output_dir, report_formats)
    _reject_catalog_report_collisions(
        catalog, report_formats=report_formats, output_dir=output_dir, basename=report_basename("dedup")
    )
    results = run_dedup_scan(
        skill_path,
        threshold=threshold,
        llm_model=llm_model,
        model=model,
        **({"catalog": catalog, "llm_verdict": llm_verdict} if catalog is not None or llm_verdict else {}),
    )
    sanitize_tier2_results(results, skill_path, catalog)
    if not emit_reports(
        results,
        report_formats=report_formats,
        output_dir=output_dir,
        basename=report_basename("dedup"),
    ):
        raise click.ClickException("dedup scan failed")


def _parse_environment_kwargs_cli(
    raw_kwargs: tuple[str, ...],
    *,
    env_mode: str | None = None,
) -> dict[str, Any]:
    """Parse and validate environment kwargs for CLI commands, raising Click BadParameter on errors."""
    from skillevaluator.tier3.evals_config import parse_environment_kwarg_overrides

    try:
        return parse_environment_kwarg_overrides(raw_kwargs, env_mode=env_mode)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint=["--ek"]) from exc


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
    "--skip",
    default=None,
    metavar="CHECKS",
    callback=_validate_skip,
    help="Leave out these checks, e.g. secrets; they are reported as skipped.",
)
@click.option(
    "--llm/--no-llm", default=None, help="Include rubric, LLM security, and verification [default: when configured]."
)
@click.option("--min-score", type=click.IntRange(0, 100), default=QUALITY_DEFAULT_MIN_SCORE, show_default=True)
@click.option(
    "--quality-config",
    type=QUALITY_CONFIG_PATH,
    default=None,
    help="Override the packaged quality-default.yaml with a custom YAML file.",
)
@click.option("--pii-waivers", type=PII_WAIVERS_PATH, default=None, help=_PII_WAIVERS_HELP)
@click.option(
    "--profile",
    default=None,
    help="Validation policy: default or registry [default: default; external is a compatibility alias].",
)
@click.option(
    "--policy",
    "policy_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Custom policy YAML overlaid on top of --profile.",
)
@click.option("--previous-version", default=None, help="Previous version for version checks.")
@click.option(
    "--skillspector-baseline", type=SKILLSPECTOR_BASELINE_PATH, default=None, help=_SKILLSPECTOR_BASELINE_HELP
)
@_workflow_report_options
def _tier1_workflow(
    skill_path: Path,
    checks: str | None,
    skip: str | None,
    llm: bool | None,
    min_score: int,
    quality_config: Path | None,
    pii_waivers: Path | None,
    profile: str | None,
    policy_path: Path | None,
    previous_version: str | None,
    skillspector_baseline: Path | None,
    report_formats: tuple[str, ...],
    output_dir: Path,
) -> None:
    """Run Tier 1 static checks and configured LLM checks for one skill."""
    from skillevaluator.tier_workflows import run_tier1_workflow
    from skillevaluator.validators.policy import resolve_policy

    try:
        policy = resolve_policy(profile=profile, policy_path=policy_path)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    _offer_scanner_install(checks, skip)
    results = run_tier1_workflow(
        skill_path,
        checks=checks,
        skip=skip,
        llm=llm,
        min_score=min_score,
        quality_config=quality_config,
        pii_waivers=pii_waivers,
        profile=policy,
        previous_version=previous_version,
        skillspector_baseline=skillspector_baseline,
    )
    # Resolved after the run so configuration errors still leave no output.
    output_dir = _resolve_file_report_output_location(skill_path.resolve(), output_dir, report_formats)
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
    help="Saved JSON catalog for inter-skill comparison.",
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
    help="Inter-skill similarity threshold.",
)
@click.option("--full-body", is_flag=True, help="Compare full SKILL.md content against a full-body catalog.")
@_workflow_report_options
def _tier2_workflow(
    skill_path: Path,
    catalog: Path | None,
    threshold: float,
    similarity_threshold: float,
    full_body: bool,
    report_formats: tuple[str, ...],
    output_dir: Path,
) -> None:
    """Run intra-skill deduplication and optional inter-skill catalog comparison."""
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
    )
    sanitize_tier2_results(results, skill_path, catalog)
    # Resolved after the run so configuration errors still leave no output; a
    # relocated directory gets the same catalog-collision check before writing.
    resolved_output_dir = _resolve_file_report_output_location(skill_path.resolve(), output_dir, report_formats)
    if resolved_output_dir != output_dir:
        _reject_catalog_report_collisions(
            catalog,
            report_formats=report_formats,
            output_dir=resolved_output_dir,
            basename=basename,
        )
    if not emit_reports(
        results,
        report_formats=report_formats,
        output_dir=resolved_output_dir,
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
        "Comma-separated Harbor agents (claude aliases claude-code; hermes is experimental, Docker/NVIDIA Build only). Default follows the provider: "
        "NVIDIA Build=opencode, OpenAI=codex, Anthropic=claude-code."
    ),
)
@click.option("--env-mode", default="docker", show_default=True, type=ENV_MODE_CHOICE)
@click.option(
    "--backend",
    default="harbor",
    show_default=True,
    type=click.Choice(["harbor", "nemo-gym"]),
    help="Tier 3 execution backend: Harbor, or NeMo Gym's orchestrator, agent harness and sandbox.",
)
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
@click.option(
    "--reuse-baseline", metavar="RUN", default=None, help="Reuse a compatible retained baseline run id or latest."
)
@click.option(
    "--compare-to",
    metavar="RUN",
    default=None,
    help="Compare with-skill scores to a retained skill version or latest.",
)
@click.option(
    "--skip-with-skill",
    is_flag=True,
    help="Skip the with-skill arm and run only the without-skill baseline (no lift analysis).",
)
@click.option("--n-attempts", type=int, default=None)
@click.option(
    "--max-agent-tokens",
    type=click.IntRange(min=0),
    default=None,
    help="Maximum measured agent-model tokens.",
)
@click.option(
    "--max-judge-calls",
    type=click.IntRange(min=0),
    default=None,
    help=(
        "Maximum recorded non-salvage judge attempts; transport retries and RAGAS-internal requests are not counted "
        "separately."
    ),
)
@click.option(
    "--max-attempts-total",
    type=click.IntRange(min=0),
    default=None,
    help="Maximum logical Harbor attempts across agents and arms, including private comparison arms.",
)
@click.option(
    "--max-cost-usd",
    type=click.FloatRange(min=0),
    default=None,
    help="Maximum agent model spend; requires --price-file and excludes judge spend, setup, and container time.",
)
@click.option(
    "--price-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Model price JSON with input_per_mtok/output_per_mtok rates and optional cached_input_per_mtok.",
)
@click.option("--pass-threshold", type=float, default=None)
@click.option(
    "--stop-on-pass/--no-stop-on-pass",
    default=None,
    help="Stop a case's remaining attempts once one passes.",
)
@click.option("--n-concurrent", type=int, default=None)
@click.option("--max-agents", type=int, default=None)
@click.option(
    "--max-cases",
    type=click.IntRange(min=1),
    default=None,
    help="Run only the first N eval cases (default: every case); a partial run is not publication evidence.",
)
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
@click.option(
    "--tolerate-baseline-timeouts/--no-tolerate-baseline-timeouts",
    default=None,
    help=(
        "Keep an agent scored on its clean with-skill arm when only baseline attempts hit the agent "
        "execution timeout; its baseline and uplift are not reported [default: disabled]."
    ),
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
@click.option(
    "--progress-interval",
    type=float,
    default=None,
    metavar="SECONDS",
    callback=_validate_progress_interval,
    help="Seconds between Tier 3 progress heartbeats (defaults to 1 for Rich and 60 for plain/CI output).",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Also print per-evaluator and per-dimension tables and the metric findings report.",
)
def evaluate(
    skill_path: Path,
    agents: str | None,
    env_mode: str,
    backend: str,
    environment_kwarg: tuple[str, ...],
    autopilot: bool,
    skip_baseline: bool,
    skip_with_skill: bool,
    reuse_baseline: str | None,
    compare_to: str | None,
    n_attempts: int | None,
    max_agent_tokens: int | None,
    max_judge_calls: int | None,
    max_attempts_total: int | None,
    max_cost_usd: float | None,
    price_file: Path | None,
    pass_threshold: float | None,
    stop_on_pass: bool | None,
    n_concurrent: int | None,
    max_agents: int | None,
    max_cases: int | None,
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
    tolerate_baseline_timeouts: bool | None,
    timeout_multiplier: float | None,
    override_cpus: int | None,
    override_memory_mb: int | None,
    override_storage_mb: int | None,
    evaluated_source_repository: str | None,
    evaluated_source_revision: str | None,
    evaluator_container_revision: str | None,
    progress: str,
    progress_interval: float | None,
    verbose: bool,
    plugin_options=None,
    plugin_report_formats: tuple[str, ...] = ("cli",),
    plugin_output_dir: Path | None = None,
) -> None:
    """Run Tier 3 live agent evaluation."""
    from skillevaluator.evaluation import EvaluationOptions, EvaluationService
    from skillevaluator.tier3.harbor.progress import create_progress_reporter

    if skip_baseline and skip_with_skill:
        # Reject before --autopilot can spend a generation call on a run that cannot start.
        raise click.UsageError("--skip-baseline and --skip-with-skill are mutually exclusive (no arm would run)")
    if reuse_baseline and (skip_baseline or skip_with_skill):
        raise click.UsageError("--reuse-baseline is mutually exclusive with --skip-baseline and --skip-with-skill")
    if compare_to and skip_with_skill:
        raise click.UsageError("--compare-to is mutually exclusive with --skip-with-skill")
    from skillevaluator.tier3.evals_config import EvalsConfigError, load_evals_config

    try:
        if plugin_options is not None:
            from skillevaluator.cli_core import resolve_plugin_path
            from skillevaluator.tier3.plugin_workflow import _load_config

            merged_config, _ = _load_config(resolve_plugin_path(skill_path))
        else:
            merged_config, _ = load_evals_config(skill_path)
    except EvalsConfigError as exc:
        raise click.UsageError(str(exc)) from exc
    harbor_config = merged_config.get("harbor", {})
    effective_max_cost = max_cost_usd if max_cost_usd is not None else harbor_config.get("max_cost_usd")
    effective_price_file = price_file or harbor_config.get("price_file")
    if effective_max_cost is not None and effective_price_file is None:
        raise click.UsageError("--max-cost-usd requires --price-file")
    service = EvaluationService()
    # Resolved before the service is built so a non-canonical value is refused
    # at the boundary rather than after a long live run has already started.
    evaluated_source = _evaluated_source_from_options(
        evaluated_source_repository,
        evaluated_source_revision,
        evaluator_container_revision,
    )
    if autopilot and plugin_options is not None:
        raise click.UsageError("Author a plugin dataset or member datasets before running evaluate-plugin")
    local_runtime_offered = False

    def prepare_local_runtime() -> None:
        nonlocal local_runtime_offered
        _offer_local_runtime_install(agents)
        local_runtime_offered = True

    if autopilot:
        _ensure_autopilot_dataset(
            skill_path,
            progress=progress,
            evaluation_params=click.get_current_context().params,
            prepare_environment=prepare_local_runtime if backend == "harbor" and env_mode == "local" else None,
        )

    parsed_environment_kwargs = _parse_environment_kwargs_cli(environment_kwarg, env_mode=env_mode)

    options = EvaluationOptions(
        skill_path=skill_path,
        agents=agents,
        env_mode=env_mode,
        backend=backend,
        environment_kwarg=environment_kwarg,
        skip_baseline=skip_baseline,
        skip_with_skill=skip_with_skill,
        reuse_baseline=reuse_baseline,
        compare_to=compare_to,
        n_attempts=n_attempts,
        max_agent_tokens=max_agent_tokens,
        max_judge_calls=max_judge_calls,
        max_attempts_total=max_attempts_total,
        max_cost_usd=max_cost_usd,
        price_file=price_file,
        pass_threshold=pass_threshold,
        stop_on_pass=stop_on_pass,
        n_concurrent=n_concurrent,
        max_agents=max_agents,
        max_cases=max_cases,
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
        tolerate_baseline_timeouts=tolerate_baseline_timeouts,
        timeout_multiplier=timeout_multiplier,
        override_cpus=override_cpus,
        override_memory_mb=override_memory_mb,
        override_storage_mb=override_storage_mb,
        evaluated_source=evaluated_source,
        environment_kwargs=parsed_environment_kwargs,
    )
    try:
        if backend == "harbor" and env_mode == "local":
            if not local_runtime_offered:
                prepare_local_runtime()
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
        elif backend == "harbor" and env_mode == "gke":
            from skillevaluator.tier3.commands import parse_agents
            from skillevaluator.tier3.harbor.runner import (
                is_gke_vertex_workload_identity_active,
                is_gke_workload_identity_allowed,
            )

            if is_gke_vertex_workload_identity_active(
                env_mode, parse_agents(agents)
            ) and is_gke_workload_identity_allowed(options.environment_kwargs):
                from rich.panel import Panel
                from rich.text import Text

                console.print(
                    Panel(
                        Text(
                            "Intended for trusted skills and least-privilege service accounts. Harbor 0.25.0 single-pod "
                            "GKE execution shares the pod service account and metadata server with evaluated skill commands.",
                            style="yellow",
                        ),
                        title=Text("GKE Workload Identity · Trusted Skills Only", style="bold cyan"),
                        border_style="yellow",
                        padding=(0, 1),
                    )
                )
        progress_reporter = create_progress_reporter(
            progress,
            stream=click.get_text_stream("stderr"),
            refresh_interval=progress_interval,
        )
        engine_result = (
            service.evaluate_plugin(options, plugin_options=plugin_options, progress_reporter=progress_reporter)
            if plugin_options is not None
            else service.evaluate(options, progress_reporter=progress_reporter)
        )
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

            render_evaluation_result(display_result, console=console, verbose=verbose, skill_path=skill_path)
        if plugin_options is not None:
            from skillevaluator.cli_plugin import emit_plugin_run_reports

            emit_plugin_run_reports(
                skill_path,
                engine_result,
                report_formats=plugin_report_formats,
                output_dir=plugin_output_dir or Path("reports"),
                results_dir=results_dir,
                env_mode=env_mode,
                console=console,
            )
        if failure:
            raise click.exceptions.Exit(1)
    except (click.ClickException, click.exceptions.Exit):
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
@click.option(
    "--prompt",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Author guidance with content beyond headings (default: evals/EVAL.md).",
)
@click.option(
    "--grounding",
    type=click.Choice(["auto", "skill", "intent", "trace"]),
    default="auto",
    show_default=True,
    help="Auto requires substantive author guidance or usable matching retained traces; "
    "existing datasets need --force or --refine. Explicit trace opts into refinement; "
    "use skill for skill-only starters.",
)
@click.option(
    "--refine",
    is_flag=True,
    help="Opt into refinement using retained or collected trajectories after grounding selection.",
)
@click.option("--from-results", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--results-dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=None)
def create_dataset(
    skill_path: Path,
    full: bool,
    no_llm: bool,
    dry_run: bool,
    force: bool,
    prompt: Path | None,
    grounding: str,
    refine: bool,
    from_results: Path | None,
    results_dir: Path | None,
) -> None:
    """Create eval datasets from author guidance, retained traces, or explicit skill grounding."""
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
                grounding=grounding,
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


@cli.command("dashboard")
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--port",
    type=click.IntRange(1, 65535),
    default=None,
    help="Local port on 127.0.0.1 (default 8501, or the next free port).",
)
@click.option("--no-browser", is_flag=True, help="Start the server without opening a browser.")
def dashboard_command(paths: tuple[Path, ...], port: int | None, no_browser: bool) -> None:
    """Compare saved JSON reports or retained Tier 3 run directories in tables.

    Install skillevaluator[dashboard] first. With no PATHS, upload JSON reports
    in the browser. The server listens on 127.0.0.1; Ctrl+C stops it.
    """
    from skillevaluator.reporting.dashboard import launch_dashboard

    raise SystemExit(launch_dashboard(paths, port=port, no_browser=no_browser))


_DOCTOR_TIERS = ("1", "2", "3", "all")
# Options that apply only to some tiers. An explicit value for a tier that was
# not selected is rejected instead of being silently ignored.
_DOCTOR_OPTION_TIERS = {
    "checks": frozenset({"1"}),
    "timeout": frozenset({"1", "2"}),
    "verify_models": frozenset({"2", "3"}),
    "agents": frozenset({"3"}),
    "env_mode": frozenset({"3"}),
    "environment_kwarg": frozenset({"3"}),
    "agent_model": frozenset({"3"}),
    "backend": frozenset({"3"}),
}


def _doctor_tier_options(*, timeout_help: str):
    """Add the tier selection and Tier 1/2 options shared by doctor and health-check."""

    def decorate(func):
        func = click.option(
            "--timeout",
            default=DEFAULT_PROBE_TIMEOUT_SEC,
            show_default=True,
            type=_FiniteFloatRange(min=0, min_open=True, max=MAX_PROBE_TIMEOUT_SEC),
            help=timeout_help,
        )(func)
        func = click.option(
            "--checks",
            default=None,
            help="Tier 1 checks whose scanners to probe, named as in validate --checks (default: validate's checks).",
        )(func)
        return click.option(
            "--tier",
            type=click.Choice(_DOCTOR_TIERS),
            default="3",
            show_default=True,
            help="Readiness to check: 1 scanners, 2 embedding and LLM providers, 3 live evaluation, or all.",
        )(func)

    return decorate


def _tier3_doctor_options(func):
    """Add the Tier 3 options shared by doctor and tier3 doctor."""
    func = click.option(
        "--backend",
        default="harbor",
        show_default=True,
        type=click.Choice(["harbor", "nemo-gym"]),
        help="Tier 3 execution backend to check: Harbor, or NeMo Gym.",
    )(func)
    func = click.option("--agent-model", multiple=True, help="Per-agent model override, AGENT=MODEL.")(func)
    func = click.option(
        "--environment-kwarg",
        "--ek",
        multiple=True,
        help="Harbor environment constructor kwarg, KEY=VALUE. Repeat for multiple values; never pass secrets.",
    )(func)
    func = click.option("--env-mode", default="docker", show_default=True, type=ENV_MODE_CHOICE, metavar="MODE")(func)
    return click.option(
        "-a",
        "--agents",
        default=None,
        help=(
            "Comma-separated Harbor agents (claude aliases claude-code; hermes is experimental, Docker/NVIDIA Build only). Default follows the provider: "
            "NVIDIA Build=opencode, OpenAI=codex, Anthropic=claude-code."
        ),
    )(func)


def _run_doctor(
    ctx: click.Context,
    *,
    tier: str,
    agents: str | None,
    env_mode: str,
    environment_kwarg: tuple[str, ...],
    checks: str | None = None,
    timeout: float = DEFAULT_PROBE_TIMEOUT_SEC,
    agent_model: tuple[str, ...] = (),
    verify_models: bool = False,
    backend: str = "harbor",
) -> int:
    """Validate every option before any probe, then check each selected tier in order."""
    selected_tiers = ("1", "2", "3") if tier == "all" else (tier,)

    def explicit(name: str) -> bool:
        return ctx.get_parameter_source(name) not in (None, click.core.ParameterSource.DEFAULT)

    for name, tiers in _DOCTOR_OPTION_TIERS.items():
        if explicit(name) and tiers.isdisjoint(selected_tiers):
            raise click.UsageError(f"--{name.replace('_', '-')} does not apply to --tier {tier}.")
    if explicit("timeout") and selected_tiers == ("2",) and not verify_models:
        # Tier 2 stays offline unless --verify-models sends live requests, so
        # there is nothing for --timeout to bound.
        condition = " without --verify-models" if "verify_models" in ctx.params else ""
        raise click.UsageError(f"--timeout does not apply to --tier 2{condition}.")
    if "1" in selected_tiers:
        unknown = sorted(set(enabled_check_lineup(checks)) - RECOGNIZED_CHECKS)
        if unknown:
            raise click.BadParameter(f"unknown Tier 1 check(s): {', '.join(unknown)}", param_hint="'--checks'")
    parsed_environment_kwargs = (
        _parse_environment_kwargs_cli(environment_kwarg, env_mode=env_mode) if "3" in selected_tiers else {}
    )

    # Tiers are independent: a failed tier never hides the readiness of the next.
    exit_codes = []
    for selected in selected_tiers:
        if selected == "3":
            from skillevaluator.tier3.commands import doctor as tier3_doctor

            exit_codes.append(
                tier3_doctor(
                    agents=agents,
                    env_mode=env_mode,
                    environment_kwarg=environment_kwarg,
                    verify_models=verify_models,
                    agent_model=agent_model,
                    environment_kwargs=parsed_environment_kwargs,
                    backend=backend,
                )
            )
        else:
            from skillevaluator.doctor import doctor as lower_tier_doctor

            exit_codes.append(
                lower_tier_doctor(tier=selected, checks=checks, verify_models=verify_models, timeout=timeout)
            )
    return 1 if any(exit_codes) else 0


@cli.command()
@_doctor_tier_options(
    timeout_help="Seconds allowed for each Tier 1 scanner probe and, with --verify-models, each Tier 2 live request."
)
@_tier3_doctor_options
@click.option(
    "--verify-models",
    is_flag=True,
    help=(
        "Make live credential-bearing requests: Tier 3 checks resolved agent-model catalog reachability; "
        "Tier 2 sends one embedding and one chat request."
    ),
)
@click.pass_context
def doctor(ctx: click.Context, **options: Any) -> None:
    """Check live-evaluation readiness, or select another tier with --tier."""
    raise SystemExit(_run_doctor(ctx, **options))


@click.command("doctor", cls=cli.command_class, context_settings=CONTEXT_SETTINGS)
@_tier3_doctor_options
@click.option(
    "--verify-models",
    is_flag=True,
    help="Check resolved agent-model catalog reachability with a live credential-bearing request.",
)
@click.pass_context
def tier3_doctor_command(ctx: click.Context, **options: Any) -> None:
    """Check live-evaluation runtime readiness."""
    raise SystemExit(_run_doctor(ctx, tier="3", **options))


@cli.command("health-check")
@_doctor_tier_options(timeout_help="Seconds allowed for each Tier 1 scanner probe.")
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
@click.pass_context
def health_check(ctx: click.Context, **options: Any) -> None:
    """Quick readiness check for the CLI and selected live-eval backend, or another tier with --tier."""
    raise SystemExit(_run_doctor(ctx, **options))


@tier3.command("import-metrics")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="New directory for external-only JSON and Markdown reports; must not exist.",
)
def tier3_import_metrics(source: Path, output_dir: Path) -> None:
    """Report precomputed custom scores without executing tests or assigning standard scores."""
    from skillevaluator.reporting.external_metrics import import_external_metrics

    try:
        import_external_metrics(source, output_dir)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo("Imported external observations; SkillEvaluator did not execute these tests.")
    click.echo(f"Reports: {output_dir / 'external_metrics.json'}, {output_dir / 'external_metrics.md'}")


@tier3.command("migrate-dataset")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--skill-name", default=None, help="Target skill identity; required when it cannot be inferred unambiguously."
)
@click.option(
    "--output",
    "destination",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write to a new JSON/YAML path. Omit to preview without writing.",
)
def tier3_migrate_dataset(source: Path, skill_name: str | None, destination: Path | None) -> None:
    """Preview a lossless dataset conversion, or write an explicit new destination."""
    from skillevaluator.tier3.dataset_migration import migrate_dataset

    try:
        result = migrate_dataset(source, skill_name=skill_name, destination=destination)
    except (OSError, ValueError, TypeError) as exc:
        raise click.ClickException(str(exc)) from exc
    if destination is None:
        ensure_ascii = json_needs_ascii(result.dataset, getattr(sys.stdout, "encoding", None))
        click.echo(json.dumps(result.dataset, indent=2, ensure_ascii=ensure_ascii))
    else:
        click.echo(f"Wrote {len(result.dataset['evals'])} cases to {destination}; source preserved.")


@tier3.command("validate")
@_skill_argument
@click.option("--json", "as_json", is_flag=True, help="Emit JSON output.")
@click.option("--strict", is_flag=True, help="Treat warnings as failures.")
@click.option("--harbor-contract", is_flag=True, help="Validate Harbor task and reward contract.")
def tier3_validate(skill_path: Path, as_json: bool, strict: bool, harbor_contract: bool) -> None:
    """Validate Tier 3 evals/ and optional Harbor BYOT contract."""
    from skillevaluator.tier3.commands import validate_evals as tier3_validate_evals

    raise SystemExit(tier3_validate_evals(skill_path, as_json=as_json, strict=strict, harbor_contract=harbor_contract))


@tier3.command("compare-versions")
@_skill_argument
@click.option("--current", required=True, metavar="RUN", help="Current retained run id or latest.")
@click.option("--prior", required=True, metavar="RUN", help="Prior retained run id or latest different revision.")
@click.option("--results-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--json", "as_json", is_flag=True, help="Print the comparison as JSON.")
def compare_versions_command(
    skill_path: Path, current: str, prior: str, results_dir: Path | None, as_json: bool
) -> None:
    """Compare two retained skill versions without execution or writes."""
    from skillevaluator.tier3.results_location import resolve_results_root
    from skillevaluator.tier3.version_comparison import compare_retained_runs, version_comparison_lines

    comparison = compare_retained_runs(resolve_results_root(skill_path, results_dir), current, prior)
    if as_json:
        click.echo(json.dumps(comparison, indent=2))
    else:
        from rich.table import Table
        from rich.text import Text

        for line in version_comparison_lines(comparison):
            console.print(Text(line))
        if comparison.get("status") == "available":
            table = Table("Agent", "Prior", "Current", "Delta", "Regression")
            for agent, values in comparison["agents"].items():
                table.add_row(
                    Text(agent),
                    f"{values['prior_overall']:.4f}",
                    f"{values['current_overall']:.4f}",
                    f"{values['delta']:+.4f}",
                    str(values["regression"]),
                )
            console.print(table)
    if comparison.get("status") != "available":
        raise click.exceptions.Exit(1)


@tier3.command("history")
@_skill_argument
@click.option("--results-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--agent", default=None, help="Include only this agent's history.")
@click.option("--json", "as_json", is_flag=True, help="Print history as JSON.")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Export to .json, .csv or .md outside the results root.",
)
def tier3_history(
    skill_path: Path, results_dir: Path | None, agent: str | None, as_json: bool, output: Path | None
) -> None:
    """List retained Skill Lift runs chronologically without execution."""
    from skillevaluator.tier3.history import collect_history, export_history, print_history
    from skillevaluator.tier3.results_location import iter_candidate_results_roots

    try:
        if output is not None:
            if output.suffix.lower() not in {".json", ".csv", ".md"}:
                raise click.UsageError("--output must end in .json, .csv or .md")
            if any(
                output.resolve().is_relative_to(root.resolve())
                for root in iter_candidate_results_roots(skill_path, results_dir)
            ):
                raise click.UsageError("--output must be outside the results roots (history is read-only)")
        history = collect_history(skill_path, results_dir, agent)
        if output is not None:
            output.write_text(export_history(history, output.suffix.lower()), encoding="utf-8", newline="")
        if as_json:
            click.echo(export_history(history, ".json"), nl=False)
        else:
            print_history(history, console)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@tier3.command("feedback")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.option(
    "--skill",
    "skill_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Evaluated skill directory; inferred from the run when omitted.",
)
def tier3_feedback(run_dir: Path, skill_path: Path | None) -> None:
    """Chat about one completed run's results and possible skill improvements."""
    from skillevaluator.tier3.commands import run_feedback_chat

    raise SystemExit(run_feedback_chat(run_dir, skill_path=skill_path))


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

# The namespace registration stays visible; only the top-level duplicate is
# hidden (same underlying command, shared params and behavior).
_tier3_evaluate_visible = copy.copy(evaluate)
# The shallow copy shares the mutable params list; give the visible twin its
# own list so in-place registration on one can never leak into the other.
_tier3_evaluate_visible.params = list(evaluate.params)
_tier3_evaluate_visible.hidden = False
tier3.add_command(_tier3_evaluate_visible, "evaluate")

from skillevaluator.cli_plugin import build_plugin_commands

for _plugin_command in build_plugin_commands(
    evaluate, run_options=_plugin_run_options, report_options=_report_options, console=console
):
    tier3.add_command(_plugin_command)
tier3.add_command(create_dataset, "create-eval-dataset")
tier3.add_command(init_custom_grader, "init-custom-grader")
tier3.add_command(init_harbor_task, "init-harbor-task")
tier3.add_command(compare, "compare")
tier3.add_command(view, "view")
tier3.add_command(tier3_doctor_command, "doctor")

from skillevaluator.tier3.decoy_commands import select_decoys_command
from skillevaluator.tier3.grader_validation import validate_grader

tier3.add_command(select_decoys_command, "select-decoys")
tier3.add_command(validate_grader, "validate-grader")

from skillevaluator.tier3.interaction_commands import interact_command

tier3.add_command(interact_command, "interact")

from skillevaluator.tier3.workflow import build_tier3_workflow

tier3.workflow = build_tier3_workflow(evaluate)
cli.add_command(harbor_view, "harbor-view")

from skillevaluator.ci_onboarding import ci as ci_onboarding_group

cli.add_command(ci_onboarding_group)


if __name__ == "__main__":
    cli()
