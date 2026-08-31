# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line interface for SkillEvaluator."""

from __future__ import annotations

import copy
import logging
import math
from pathlib import Path

import click

from skillevaluator import __version__
from skillevaluator.cli_help import GroupedOption, RichGroup
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

# Tier 1 (static validation) is the base install surface and is safe to import
# eagerly. Tier 2 (embeddings/LLM) and Tier 3 (Harbor and its environments)
# pull heavy, extras-only dependencies, so their command implementations are
# imported lazily inside the command callbacks. This keeps `import skillevaluator.cli`
# and the CLI surface available on a base install without those extras.
from skillevaluator.tier1.commands import (
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
from skillevaluator.tier3_environments import HARBOR_ENVIRONMENTS
from skillevaluator.utils.tier2_paths import (
    is_link_or_reparse,
    paths_refer_to_same_location,
    sanitize_tier2_results,
)

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


# Heading + intro for the grouped Tier 3 options in ``validate --help``.
_RUN_GROUP = "Run & Reports"
_RUN_GROUP_DESC = "Applies to the whole run: target typing, policy profile, reports, tier selection."
_TIER1_GROUP = "Tier 1 · Static & Security"
_TIER1_GROUP_DESC = "Static checks; LLM-free by default. Tier 1 gates the exit code and always runs."
_TIER2_GROUP = "Tier 2 · Deduplication"
_TIER2_GROUP_DESC = "Embedding + LLM dedup; on by default, skips gracefully without a provider key."
_TIER3_GROUP = "Tier 3 · Live Agent Evaluation"
_TIER3_GROUP_DESC = "Forwarded to the live-eval engine only when --agent-eval is also passed."

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
  Tier 3  Live agent evaluation (advisory; enable with --tier3, --autopilot,
          --full, or the --agent-eval alias).

LLM analysis (Tier 1 security + Tier 3 dimension judge):
  Off by default. Configure a public LLM provider with
  SKILL_EVAL_LLM_PROVIDER and its provider credential, then add --llm.
  Add --llm-verify for a second pass that suppresses false positives.

Examples:
  skillevaluator validate ./my-skill                        # Tier 1 + Tier 2 (cli)
  skillevaluator validate ./my-skill --llm                  # add LLM security scan
  skillevaluator validate ./my-skill -r cli -r json -r html # multiple reports (repeat -r)
  skillevaluator validate ./my-skill -r cli,json,html       # comma-separated too
  skillevaluator validate ./my-skill -o reports/            # custom output dir
  skillevaluator validate ./my-skill --no-dedup             # skip Tier 2 dedup
  skillevaluator validate ./my-skill --external             # strict publish profile
  skillevaluator validate ./my-skill -c                     # continue on failure (record all issues)
  skillevaluator validate ./my-skill --tier3 -a codex      # + Tier 3 eval
  skillevaluator validate ./my-skill --autopilot            # + Tier 3, generating evals
  skillevaluator validate ./my-skill --full -a codex        # everything, one shot
  skillevaluator validate ./my-skill --tiers 1,3            # explicit tier selection
  skillevaluator validate ./skills-folder --full            # whole catalog, serially
  skillevaluator validate ./my-skill --agent-eval -a codex,claude-code \\
      --env-mode docker --harbor-keep-jobs                 # + Tier 3, retain Harbor jobs
"""


_TOP_LEVEL_COMMAND_HELP_GROUPS = (
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
    ("Expert aliases", ("tier1", "tier2", "tier3")),
)


@click.group(
    cls=RichGroup,
    context_settings=CONTEXT_SETTINGS,
    help_command_groups=_TOP_LEVEL_COMMAND_HELP_GROUPS,
)
@click.version_option(version=__version__, prog_name="skillevaluator")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def cli(verbose: bool) -> None:
    """SKILLEVALUATOR: SkillEvaluator for AI agent skills.

    Three-tier quality gatekeeper for AI agent skills and plugins.

    Documentation: https://docs.nvidia.com/skills/skillevaluator/
    """
    setup_logging(verbose=verbose)


@cli.group()
def tier1() -> None:
    """Expert aliases for Tier 1 static checks."""


@cli.group()
def tier2() -> None:
    """Expert aliases for Tier 2 similarity and deduplication checks."""


@cli.group()
def tier3() -> None:
    """Expert aliases for Tier 3 dataset creation and live agent evaluation."""


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

    def _skip(message: str) -> list[ValidationResult]:
        result = ValidationResult(
            validator_name="Tier 2 Deduplication",
            validator_description="Embedding-based duplicate detection",
        )
        result.add_warning(message)
        result.metadata["skipped"] = True
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
        return _skip(f"Skipped: configure a public embedding provider ({exc}), or pass --no-dedup.")
    return run_dedup_scan(target_path)


def _partial_agent_eval_result(
    target_path: Path,
    *,
    engine_result: object,
    failure: str,
    results_dir: Path | None,
    env_mode: str,
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


def _run_agent_eval_or_skip(
    target_path: Path,
    *,
    agents: str,
    env_mode: str,
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
    block_on_agent_eval: bool = False,
    validate_source: bool = True,
    progress_reporter=None,
) -> ValidationResult:
    """Run Tier 3 live agent evaluation and fold the result into the combined report.

    Returns an ``AGENT_EVAL`` :class:`ValidationResult` carrying the canonical
    ``metadata["agent_eval"]`` payload on success, or a structured result
    describing why Tier 3 could not run. Tier 3 remains advisory by default,
    and callers can opt into blocking behavior.
    """
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


# Per-tier section headings printed by ``validate`` as each tier runs. They give
# the CLI/CI stream the same progressive, labeled structure SkillEvaluator emitted, so
# Tier 1 (and Tier 2) are visibly reported as they execute instead of only
# surfacing in the single combined report rendered at the very end.
_TIER_BANNERS = {
    "tier1": "Tier 1: Security and Static Validation",
    "tier2": "Tier 2: Deduplication",
    "tier3": "Tier 3: Live Agent Evaluation",
}


def _ensure_autopilot_dataset(skill_path: Path, *, quiet: bool = False) -> str | None:
    """Ensure an evaluation source exists, generating one when missing.

    Mirrors the standalone ``evaluate --autopilot`` behavior: reuse an existing
    source unchanged; otherwise generate one case with the configured provider,
    falling back to a deterministic case when no provider key is available.
    Returns a short note describing what happened (for the pipeline view), or
    raises ``click.ClickException`` when generation cannot produce a source.
    """
    from skillevaluator.evaluation import EvaluationService
    from skillevaluator.provider_config import ProviderConfigurationError, resolve_llm_provider
    from skillevaluator.tier3.harbor.adapter import find_evals_file

    def echo(message: str, *, err: bool = False) -> None:
        if not quiet:
            click.echo(message, err=err)

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
            echo(f"Autopilot: generating one case with the configured {provider.provider} provider.")

        try:
            service.create_autopilot_dataset(skill_path, use_llm=not no_llm)
        except FileExistsError:
            echo("Autopilot: an evaluation source appeared concurrently; reusing it unchanged.")
        except Exception:
            if no_llm:
                raise
            echo("Warning: Autopilot LLM generation failed; falling back to one deterministic case.", err=True)
            if not eval_source_exists():
                service.create_autopilot_dataset(skill_path, use_llm=False)
        if not eval_source_exists():
            raise click.ClickException("Autopilot dataset generation did not produce an evaluation source.")
        return "auto-generated by autopilot — review evals/"
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
        (name, "bold"),
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
            lines.append(Text.assemble(("  ✗ ", f"bold {RED}"), (f"{name:<28}", TEXT), (reason[:56], MUTED)))
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
    console_.print(Text.assemble(("      reports     ", MUTED), (f"{reports_root}/<skill>/", MUTED)))


def _validate_catalog(
    ctx: click.Context,
    *,
    resolved_target: Path,
    output_dir: Path,
) -> None:
    """Run the full validate pipeline once per skill in the catalog, serially.

    Each skill is an independent job with its own pipeline view, reports
    (under ``<output_dir>/<skill>/``), and verdict; the catalog exits nonzero
    when any skill failed.
    """
    if ctx.params.get("previous_version"):
        raise click.ClickException(
            "--previous-version applies to one skill and cannot be reused for a catalog; "
            "validate each skill separately with its own previous version"
        )

    skill_dirs = sorted(marker.parent for marker in resolved_target.glob("*/SKILL.md"))
    failures: list[tuple[str, str]] = []
    for index, skill_dir in enumerate(skill_dirs, start=1):
        _print_catalog_divider(index, len(skill_dirs), skill_dir.name)
        overrides = {
            **ctx.params,
            "target_path": skill_dir,
            "content_type": "skill",
            "output_dir": output_dir / skill_dir.name,
        }
        try:
            ctx.invoke(validate, **overrides)
        except click.ClickException as exc:
            failures.append((skill_dir.name, str(getattr(exc, "message", exc))))
        except Exception as exc:  # unexpected: keep the catalog running, report it on the scoreboard
            failures.append((skill_dir.name, f"unexpected error: {exc}"))
    _print_catalog_summary(len(skill_dirs), failures, output_dir)
    if failures:
        raise click.ClickException(
            f"{len(failures)}/{len(skill_dirs)} skills failed validation: "
            + ", ".join(name for name, _reason in failures)
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

    if not gate_failed:
        ran = sum(1 for block in view.blocks if block.status not in ("pending", "skip"))
        advisory_failed = any(block.status == "fail" and block.number in {2, 3} for block in view.blocks)
        if advisory_failed:
            headline = "gating tiers passed · advisory tier reported findings (see report)"
        else:
            headline = f"all {ran} tier{'s' if ran != 1 else ''} passed"
            lift = summary.get("overall_lift")
            if agent_eval and isinstance(lift, (int, float)):
                headline += f"  ·  skill lift {lift:+.2f}"
        view.finish(Verdict(passed=True, headline=headline), links)
        return

    failed = [result for result in tier_gate_results if not result.passed and not _is_skipped(result)]
    if failed:
        first = failed[0].validator_name or "validation"
        extra = f" (+{len(failed) - 1} more)" if len(failed) > 1 else ""
        tier_no = (failed[0].metadata.get("gating") or {}).get("tier")
        tier_label = f" in Tier {tier_no}" if tier_no is not None else ""
        headline = f"{first}{extra} failed{tier_label}"
    else:
        headline = "validation failed"
    rerun = _rerun_hint(target_path, agent_eval)
    view.finish(
        Verdict(passed=False, headline=headline, fix=first_fix(tier_gate_results), rerun=rerun),
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
    console.print(f"Target: {target_path}")
    console.print(f"Type: {content_type}")
    if profile:
        profile_color = "cyan"
        console.print(f"Profile: [{profile_color}]{profile}[/{profile_color}]")


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
    help="One-shot validation: Tier 1+2+3 with --autopilot dataset generation.",
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
    "opt-in (not run by default): dependency. "
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
    "--tier3",
    "--agent-eval",
    "agent_eval",
    is_flag=True,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Also run Tier 3 live agent evaluation (requires a valid eval dataset or native Harbor source).",
)
@click.option(
    "--block-on-agent-eval/--no-block-on-agent-eval",
    default=None,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Make Tier 3 findings gate the exit code. Default: advisory.",
)
@click.option(
    "--autopilot",
    is_flag=True,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Generate an evaluation source automatically when missing, then run Tier 3 (implies --tier3).",
)
@click.option(
    "-a",
    "--agents",
    default="codex",
    show_default=True,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Comma-separated Harbor agents to evaluate.",
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
    dedup: bool,
    block_on_dedup: bool | None,
    agent_eval: bool,
    block_on_agent_eval: bool | None,
    autopilot: bool,
    agents: str,
    env_mode: str,
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
    report_formats: tuple[str, ...],
    output_dir: Path,
) -> None:
    """Validate a skill, rule, workflow, or plugin (Tier 1, with optional Tier 2/Tier 3).

    Runs Tier 1 static, security, and quality checks (which gate the exit code),
    plus blocking Tier 2 deduplication by default. Add --agent-eval for
    advisory Tier 3 live evaluation, or --block-on-agent-eval to make it gate.
    Reports are written per --report and --output-dir.

    A plugin (a bundle-reference ``agent_plugin.yaml``/``.yml`` manifest or a
    contained ``.claude-plugin/plugin.json`` manifest) is auto-detected and
    validated against its public contract. Quality/lint/version checks are
    skill-only and skipped for plugins.
    """
    if dedup:
        _reject_linked_tier2_root(target_path)
    target_path = target_path.resolve()

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

    if external and profile and profile != "external":
        raise click.ClickException(f"--external conflicts with --profile {profile}; pass one or the other.")
    profile_name = "external" if external else profile
    try:
        policy = resolve_policy(profile=profile_name, policy_path=policy_path)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    block_on_dedup_effective = True if block_on_dedup is None else block_on_dedup
    block_on_agent_eval_effective = False if block_on_agent_eval is None else block_on_agent_eval

    resolved_type = content_type if content_type != "auto" else detect_content_type(target_path)
    # Only skills own an evals/ task source. Plugins, rules, and workflows can
    # still run Tier 3, but must bypass the skill-directory source preflight.
    preflight_tier3_source = resolved_type == CONTENT_TYPE_SKILL

    # --full is the one-shot (everything incl. autopilot); --autopilot implies
    # Tier 3; --tiers is the explicit selector. Explicit --no-tier2 still wins.
    if full:
        autopilot = True
    if autopilot:
        agent_eval = True
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
        if not agent_eval:
            autopilot = False

    from skillevaluator.constants import CONTENT_TYPE_UNKNOWN

    # A directory of skills (no root SKILL.md) is a catalog: run the pipeline
    # once per skill, serially, each as its own job with its own reports.
    if (
        resolved_type in (CONTENT_TYPE_SKILL, CONTENT_TYPE_UNKNOWN)
        and target_path.is_dir()
        and not (target_path / "SKILL.md").exists()
        and any(target_path.glob("*/SKILL.md"))
    ):
        _validate_catalog(
            click.get_current_context(),
            resolved_target=target_path,
            output_dir=output_dir,
        )
        return

    # Quiet (default) drives the compact pipeline view; --verbose keeps the
    # historical full-detail stream, as does DEBUG logging via the group -v.
    quiet = not verbose and not logging.getLogger().isEnabledFor(logging.DEBUG)
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
        command="validate --tier3" if agent_eval else "validate",
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
        target_path,
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
    )
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

    if dedup and not (fail_fast and not continue_on_failure and tier1_raw_failed):
        if not quiet:
            _print_tier_banner(_TIER_BANNERS["tier2"])
        view.tier_start(tier2_index)
        view.tier_progress(tier2_index, [stage_hint_row("stages", "chunk · embed · cluster · llm-judge")])
        tier2_results = _run_dedup_or_skip(target_path)
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
            "local": "experimental host sandbox — trusted skills and workspaces only",
        }.get(env_mode, "")
        model_display = ", ".join(agent_model) if agent_model else (model or "agent defaults")
        tier3_config_rows = [
            detail_row("agent", agents),
            detail_row("env", env_mode, env_note),
            detail_row("model", model_display),
        ]

        # Autopilot: reuse the standalone evaluate command's dataset flow.
        # Tier 3 is advisory, so a dataset-generation failure must not abort
        # validate after Tier 1/2 already ran -- Tier 3 skips with the reason.
        autopilot_error: str | None = None
        if autopilot:
            try:
                dataset_note = _ensure_autopilot_dataset(target_path, quiet=quiet)
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
            target_path,
            agents=agents,
            env_mode=env_mode,
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
            block_on_agent_eval=block_on_agent_eval_effective,
            validate_source=preflight_tier3_source,
            progress_reporter=reporter,
        )
        results.append(tier3_result)
        tier3_ran, tier3_ok, tier3_rows, tier3_skip = summarize_tier3(tier3_result)
        if autopilot_error and not tier3_ran:
            tier3_skip = f"{autopilot_error}; {tier3_skip}"
            # Reports read the skip reason from metadata, so the generation
            # failure must land there too, not only in the view's skip row.
            tier3_result.metadata["skip_reason"] = tier3_skip
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

            source_kind, _source_checks = validate_tier3_source(target_path)
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
    if quiet and not _report_formats_explicit():
        effective_formats = tuple(dict.fromkeys([f for f in report_formats if f != "cli"] + ["html", "json"]))
    else:
        effective_formats = report_formats
    report_basename_value = make_timestamped_basename(f"{REPORT_PREFIX}-output")
    emit_reports(
        results,
        report_formats=effective_formats,
        output_dir=output_dir,
        basename=report_basename_value,
        policy=policy,
        target_path=target_display,
        content_label=content_label,
        announce_paths=not quiet,
        sarif_scan_root=target_path,
        sarif_repository_root=sarif_repository_root,
    )

    # BENCHMARK.md is generated compulsorily for skills (matches SkillEvaluator), even on
    # failure, so the publication card always reflects the latest evaluation --
    # now including Tier 3 results when --agent-eval ran.
    if resolved_type == CONTENT_TYPE_SKILL:
        from skillevaluator.reporting import BenchmarkReporter
        from skillevaluator.reporting.naming import BENCHMARK_FILENAME

        output_dir.mkdir(parents=True, exist_ok=True)
        BenchmarkReporter(skill_name=target_path.name).save(results, output_dir / BENCHMARK_FILENAME)

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
            target_path=target_path,
            agent_eval=agent_eval,
        )
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
@click.option("--type", "content_type", default="auto", type=click.Choice(["skill", "rules", "workflows", "auto"]))
@click.option("--threshold", type=float, default=0.75, show_default=True, callback=_validate_similarity_threshold)
@click.option("--full-body", is_flag=True, help="Embed full file bodies instead of descriptions.")
@click.option("--model", default=None, help="Embedding model override.")
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
    help="Build and save a versioned local catalog from this collection.",
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


# Hidden top-level spelling of ``tier3 evaluate`` — kept working for scripts,
# but the tier namespace is the advertised name to avoid a duplicate surface.
@cli.command(hidden=True)
@_skill_argument
@click.option(
    "-a",
    "--agents",
    default="codex",
    show_default=True,
    help="Comma-separated Harbor agents (claude is an alias for claude-code).",
)
@click.option("--env-mode", default="docker", show_default=True, type=ENV_MODE_CHOICE)
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
    help="Run one real agent smoke task before the full evaluation matrix [default: enabled].",
)
@click.option("--timeout-multiplier", type=float, default=None)
@click.option("--override-cpus", type=int, default=None)
@click.option("--override-memory-mb", type=int, default=None)
@click.option("--override-storage-mb", type=int, default=None)
@click.option(
    "--progress",
    type=click.Choice(["auto", "rich", "plain", "off"]),
    default="auto",
    show_default=True,
    help="Tier 3 progress presentation (auto uses Rich on a TTY and plain lines otherwise).",
)
def evaluate(
    skill_path: Path,
    agents: str,
    env_mode: str,
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
    progress: str,
) -> None:
    """Run Tier 3 live agent evaluation."""
    from skillevaluator.evaluation import EvaluationOptions, EvaluationService
    from skillevaluator.tier3.harbor.progress import create_progress_reporter

    service = EvaluationService()
    if autopilot:
        _ensure_autopilot_dataset(skill_path)

    options = EvaluationOptions(
        skill_path=skill_path,
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


@cli.command("create-eval-dataset")
@_skill_argument
@click.option("--full", is_flag=True, help="Generate the full 4-bucket dataset.")
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
    default="codex",
    show_default=True,
    help="Comma-separated Harbor agents (claude is an alias for claude-code).",
)
@click.option("--env-mode", default="docker", show_default=True, type=ENV_MODE_CHOICE)
@click.option("--agent-model", multiple=True, help="Per-agent model override, AGENT=MODEL.")
@click.option(
    "--verify-models",
    is_flag=True,
    help="Check resolved agent-model catalog reachability with a live credential-bearing request.",
)
def doctor(agents: str, env_mode: str, agent_model: tuple[str, ...], verify_models: bool) -> None:
    """Check live-evaluation runtime readiness."""
    from skillevaluator.tier3.commands import doctor as tier3_doctor

    raise SystemExit(
        tier3_doctor(
            agents=agents,
            env_mode=env_mode,
            verify_models=verify_models,
            agent_model=agent_model,
        )
    )


@cli.command("health-check")
@click.option("-a", "--agents", default="codex", show_default=True)
@click.option("--env-mode", default="docker", show_default=True, type=ENV_MODE_CHOICE)
def health_check(agents: str, env_mode: str) -> None:
    """Quick readiness check for the CLI and selected live-eval backend."""
    from skillevaluator.tier3.commands import doctor as tier3_doctor

    raise SystemExit(tier3_doctor(agents=agents, env_mode=env_mode, verify_models=False, agent_model=()))


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
tier1.add_command(validate, "validate")
tier1.add_command(quality_check, "quality-check")
tier1.add_command(rubric_eval, "rubric-eval")
tier1.add_command(security_scan, "security-scan")
tier1.add_command(pii_scan, "pii-scan")
tier1.add_command(lint_scripts, "lint-scripts")

tier2.add_command(similarity_check, "similarity-check")
tier2.add_command(context_optimization_check, "context-optimization-check")
tier2.add_command(dedup_scan, "dedup-scan")

# The namespace registration stays visible; only the top-level duplicate is
# hidden (same underlying command, shared params and behavior).
_tier3_evaluate_visible = copy.copy(evaluate)
# The shallow copy shares the mutable params list; give the visible twin its
# own list so in-place registration on one can never leak into the other.
_tier3_evaluate_visible.params = list(evaluate.params)
_tier3_evaluate_visible.hidden = False
tier3.add_command(_tier3_evaluate_visible, "evaluate")
tier3.add_command(create_dataset, "create-eval-dataset")
tier3.add_command(init_custom_grader, "init-custom-grader")
tier3.add_command(init_harbor_task, "init-harbor-task")
tier3.add_command(compare, "compare")
tier3.add_command(view, "view")
tier3.add_command(doctor, "doctor")
cli.add_command(harbor_view, "harbor-view")


if __name__ == "__main__":
    cli()
