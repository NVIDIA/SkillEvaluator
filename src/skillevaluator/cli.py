# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line interface for Skill Evaluator."""

from __future__ import annotations

import math
from pathlib import Path

import click

from skillevaluator import __version__
from skillevaluator.cli_help import GroupedOption, RichGroup
from skillevaluator.logging_config import setup_logging
from skillevaluator.models.result import ValidationResult
from skillevaluator.reporting.naming import report_basename

# Tier 1 (static validation) is the base install surface and is safe to import
# eagerly. Tier 2 (embeddings/LLM) and Tier 3 (Harbor and its environments)
# pull heavy, extras-only dependencies, so their command implementations are
# imported lazily inside the command callbacks. This keeps `import skillevaluator.cli`
# and the CLI surface available on a base install without those extras.
from skillevaluator.tier1.commands import (
    console,
    emit_reports,
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
GRADING_MODE_CHOICE = click.Choice(["default", "default_plus_custom", "custom_only"])
CUSTOM_GRADING_MODE_CHOICE = click.Choice(["default_plus_custom", "custom_only"])


def _validate_similarity_threshold(_ctx: click.Context, _param: click.Parameter, value: float) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise click.BadParameter("must be finite and within [0, 1]")
    return value


# Heading + intro for the grouped Tier 3 options in ``validate --help``.
_TIER3_GROUP = "Tier 3: Live Agent Evaluation"
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
  Tier 3  Live agent evaluation (advisory; enable with --agent-eval).

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
  skillevaluator validate ./my-skill --agent-eval -a codex  # + Tier 3 eval
  skillevaluator validate ./my-skill --agent-eval -a codex,claude-code \\
      --env-mode docker --harbor-keep-jobs                 # + Tier 3, retain Harbor jobs
"""


@click.group(cls=RichGroup, context_settings=CONTEXT_SETTINGS)
@click.version_option(version=__version__, prog_name="skillevaluator")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def cli(verbose: bool) -> None:
    """SKILLEVALUATOR: SkillEvaluator for AI agent skills.

    Three-tier quality gatekeeper for AI agent skills and plugins.

    For setup and usage examples, see: README.md
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
        type=click.Choice(["cli", "json", "html", "markdown"]),
        default=("cli",),
        show_default=True,
        help="Report format(s). Accepts comma- or space-separated values "
        "(-r cli,json,html or -r cli json html) and may be repeated.",
    )(func)


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
    model: str | None = None,
    agent_model: tuple[str, ...] = (),
    grading_mode: str | None = None,
    results_dir: Path | None = None,
    include_skills: tuple[Path, ...] = (),
    copy_repo: bool = False,
    timeout_multiplier: float | None = None,
    harbor_keep_jobs: bool = False,
) -> ValidationResult:
    """Run Tier 3 live agent evaluation and fold the result into the combined report.

    Returns an ``AGENT_EVAL`` :class:`ValidationResult` carrying the canonical
    ``metadata["agent_eval"]`` payload on success, or a non-blocking advisory
    result describing why Tier 3 could not run. Tier 3 is always advisory: it is
    reported in the combined HTML/JSON/BENCHMARK.md but never gates the
    ``validate`` exit code.
    """
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
        engine_result = service.evaluate(options)
    except Exception as exc:
        # Tier 3 is advisory: degrade any evaluation error to a non-blocking
        # note rather than aborting the whole validate pipeline.
        return advisory_skip_result(
            f"Tier 3 live evaluation skipped: {exc}",
            skill_name=target_path.name,
        )

    if failure := service.failure_reason(engine_result):
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
# the CLI/CI stream the same progressive, labeled structure Skill Evaluator emitted, so
# Tier 1 (and Tier 2) are visibly reported as they execute instead of only
# surfacing in the single combined report rendered at the very end.
_TIER_BANNERS = {
    "tier1": "Tier 1: Security and Static Validation",
    "tier2": "Tier 2: Deduplication",
    "tier3": "Tier 3: Live Agent Evaluation",
}


def _print_tier_banner(title: str) -> None:
    """Print a labeled per-tier section banner (parity with Skill Evaluator)."""
    click.echo(f"\n{title}")
    click.echo("-" * 50)


def _print_run_banner(target_path: Path, content_type: str, profile: str | None) -> None:
    """Print the pre-run header (target + detected type + active profile).

    Restores parity with Skill Evaluator's ``_print_validation_banner``: before any
    tier runs, surface what is being validated, the resolved content type, and
    the active validation profile so CI logs and terminal sessions identify the
    run up front instead of opening straight on the Tier 1 section.
    """
    console.print(f"\n[bold]Skill Evaluator {content_type.title()} Validation[/bold]")
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
    help="Force the content type instead of auto-detecting it from the target path.",
)
@click.option(
    "--checks",
    help="Comma-separated subset of Tier 1 checks to run (default: all applicable). "
    "Choices: schema, security, pii, license, code-integrity, unicode, quality, lint; "
    "opt-in (not run by default): version, dependency. "
    "quality/lint/version are skill-only and skipped for rules/workflows.",
)
@click.option("--fail-fast", is_flag=True, help="Stop on the first failing check instead of collecting all issues.")
@click.option(
    "-c",
    "--continue-on-failure",
    "continue_on_failure",
    is_flag=True,
    help="Run the full pipeline without stopping early; record all issues in the reports. "
    "Overrides --fail-fast, and for folder validation keeps scanning every skill past a "
    "CRITICAL finding.",
)
@click.option(
    "--llm/--no-llm",
    default=False,
    show_default=True,
    help="Enable LLM-backed security analysis (requires a configured public provider).",
)
@click.option("--llm-verify", is_flag=True, help="Run a second LLM pass to suppress false-positive findings.")
@click.option(
    "--min-score",
    type=int,
    default=70,
    show_default=True,
    help="Minimum quality score (0-100) required to pass when the 'quality' check runs.",
)
@click.option(
    "--profile",
    default=None,
    help="Validation profile: external or a custom name. Default: $SKILLEVALUATOR_PROFILE env var, then external.",
)
@click.option("--external", is_flag=True, help="Shortcut for --profile external (validate for public publication).")
@click.option(
    "--policy",
    "policy_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Custom policy YAML overlaid on top of --profile.",
)
@click.option(
    "--dedup/--no-dedup",
    default=True,
    show_default=True,
    help="Run Tier 2 intra-skill semantic-overlap checks. On by default; skipped "
    "gracefully without public embedding access. Use --no-dedup to disable.",
)
@click.option(
    "--agent-eval",
    is_flag=True,
    cls=GroupedOption,
    help_group=_TIER3_GROUP,
    help="Also run Tier 3 live agent evaluation (requires evals/evals.json).",
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
    checks: str | None,
    fail_fast: bool,
    continue_on_failure: bool,
    llm: bool,
    llm_verify: bool,
    min_score: int,
    profile: str | None,
    external: bool,
    policy_path: Path | None,
    dedup: bool,
    agent_eval: bool,
    agents: str,
    env_mode: str,
    skip_baseline: bool,
    n_concurrent: int | None,
    max_agents: int | None,
    n_attempts: int | None,
    pass_threshold: float | None,
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
    plus Tier 2 deduplication by default. Add --agent-eval for advisory Tier 3
    live agent evaluation. Reports are written per --report and --output-dir.

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
    from skillevaluator.utils.helpers import make_timestamped_basename, resolve_git_remote_url
    from skillevaluator.validators.policy import apply_policy, resolve_policy

    if external and profile and profile != "external":
        raise click.ClickException(f"--external conflicts with --profile {profile}; pass one or the other.")
    profile_name = "external" if external else profile
    try:
        policy = resolve_policy(profile=profile_name, policy_path=policy_path)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    resolved_type = content_type if content_type != "auto" else detect_content_type(target_path)

    _print_run_banner(target_path, resolved_type, getattr(policy, "profile", None))
    _print_tier_banner(_TIER_BANNERS["tier1"])
    results = run_validation(
        target_path,
        checks=checks,
        use_llm=llm,
        llm_verify=llm_verify,
        min_score=min_score,
        policy=policy,
        content_type=resolved_type,
        fail_fast=fail_fast,
        continue_on_failure=continue_on_failure,
    )
    if dedup and not (fail_fast and not continue_on_failure and any(not r.passed for r in results)):
        _print_tier_banner(_TIER_BANNERS["tier2"])
        results.extend(_run_dedup_or_skip(target_path))

    # Tier 1 (and Tier 2) gate the exit code; Tier 3 is advisory. Snapshot the
    # gating set before Tier 3 is appended so a non-PASS live-eval verdict is
    # reported in the combined report but never changes the exit code
    # emit_reports applies the policy in
    # place, so these same objects carry the finalized pass/fail afterward.
    tier_gate_results = list(results)

    # Flush Tier 1 + Tier 2 results to the terminal BEFORE the long-running
    # Tier 3 agent evaluation so they stay visible in CI logs even when Tier 3
    # is slow, errors, or is interrupted before the combined report is emitted.
    # Severities are finalized first so this interim view matches the combined
    # report rendered at the end (apply_policy is idempotent, so emit_reports
    # re-applying it is a no-op).
    if agent_eval and "cli" in report_formats:
        apply_policy(tier_gate_results, policy)
        CLIReporter(console=console).print_summary(tier_gate_results)

    # Tier 3 runs BEFORE report emission so its results are folded into the
    # single combined HTML/JSON/BENCHMARK.md report (parity with Skill Evaluator), and
    # runs regardless of Tier 1/Tier 2 outcome. It degrades to a non-blocking
    # advisory note when it cannot run.
    if agent_eval:
        _print_tier_banner(_TIER_BANNERS["tier3"])
        results.append(
            _run_agent_eval_or_skip(
                target_path,
                agents=agents,
                env_mode=env_mode,
                skip_baseline=skip_baseline,
                n_concurrent=n_concurrent,
                max_agents=max_agents,
                n_attempts=n_attempts,
                pass_threshold=pass_threshold,
                model=model,
                agent_model=agent_model,
                grading_mode=grading_mode,
                results_dir=results_dir,
                include_skills=include_skills,
                copy_repo=copy_repo,
                timeout_multiplier=timeout_multiplier,
                harbor_keep_jobs=harbor_keep_jobs,
            )
        )

    content_label = {
        CONTENT_TYPE_SKILL: "Skill",
        CONTENT_TYPE_RULES: "Rule",
        CONTENT_TYPE_WORKFLOWS: "Workflow",
        CONTENT_TYPE_PLUGIN: "Plugin",
    }.get(resolved_type, "Skill")
    target_display = resolve_git_remote_url(target_path) or str(target_path)

    emit_reports(
        results,
        report_formats=report_formats,
        output_dir=output_dir,
        basename=make_timestamped_basename(f"{REPORT_PREFIX}-output"),
        policy=policy,
        target_path=target_display,
        content_label=content_label,
    )

    # BENCHMARK.md is generated compulsorily for skills (matches Skill Evaluator), even on
    # failure, so the publication card always reflects the latest evaluation --
    # now including Tier 3 results when --agent-eval ran.
    if resolved_type == CONTENT_TYPE_SKILL:
        from skillevaluator.reporting import BenchmarkReporter
        from skillevaluator.reporting.naming import BENCHMARK_FILENAME

        output_dir.mkdir(parents=True, exist_ok=True)
        BenchmarkReporter().save(results, output_dir / BENCHMARK_FILENAME)

    if not all(r.passed for r in tier_gate_results):
        raise click.ClickException("validation failed")


# Intro text shown under the grouped Tier 3 options in ``validate --help``.
validate.help_group_descriptions = {_TIER3_GROUP: _TIER3_GROUP_DESC}


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


@cli.command()
@_skill_argument
@click.option("-a", "--agents", default="codex", show_default=True, help="Comma-separated Harbor agents.")
@click.option("--env-mode", default="docker", show_default=True, type=ENV_MODE_CHOICE)
@click.option("--skip-baseline", is_flag=True, help="Skip without-skill baseline.")
@click.option("--n-attempts", type=int, default=None)
@click.option("--pass-threshold", type=float, default=None)
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
@click.option("--timeout-multiplier", type=float, default=None)
@click.option("--override-cpus", type=int, default=None)
@click.option("--override-memory-mb", type=int, default=None)
@click.option("--override-storage-mb", type=int, default=None)
def evaluate(
    skill_path: Path,
    agents: str,
    env_mode: str,
    skip_baseline: bool,
    n_attempts: int | None,
    pass_threshold: float | None,
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
    timeout_multiplier: float | None,
    override_cpus: int | None,
    override_memory_mb: int | None,
    override_storage_mb: int | None,
) -> None:
    """Run Tier 3 live agent evaluation."""
    from skillevaluator.evaluation import EvaluationOptions, EvaluationService

    options = EvaluationOptions(
        skill_path=skill_path,
        agents=agents,
        env_mode=env_mode,
        skip_baseline=skip_baseline,
        n_attempts=n_attempts,
        pass_threshold=pass_threshold,
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
        timeout_multiplier=timeout_multiplier,
        override_cpus=override_cpus,
        override_memory_mb=override_memory_mb,
        override_storage_mb=override_storage_mb,
    )
    try:
        service = EvaluationService()
        engine_result = service.evaluate(options)
        if failure := service.failure_reason(engine_result):
            raise click.ClickException(failure)
    except click.ClickException:
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
    from skillevaluator.evaluation import DatasetOptions, EvaluationService

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


@cli.command()
@click.option("-a", "--agents", default="codex", show_default=True)
@click.option("--env-mode", default="docker", show_default=True, type=ENV_MODE_CHOICE)
@click.option("--verify-models", is_flag=True, help="Show the configured public provider model.")
def doctor(agents: str, env_mode: str, verify_models: bool) -> None:
    """Check live-evaluation runtime readiness."""
    from skillevaluator.tier3.commands import doctor as tier3_doctor

    raise SystemExit(
        tier3_doctor(
            agents=agents,
            env_mode=env_mode,
            verify_models=verify_models,
        )
    )


@cli.command("health-check")
@click.option("-a", "--agents", default="codex", show_default=True)
@click.option("--env-mode", default="docker", show_default=True, type=ENV_MODE_CHOICE)
def health_check(agents: str, env_mode: str) -> None:
    """Quick readiness check for the CLI and selected live-eval backend."""
    from skillevaluator.tier3.commands import doctor as tier3_doctor

    raise SystemExit(tier3_doctor(agents=agents, env_mode=env_mode, verify_models=False))


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

tier3.add_command(evaluate, "evaluate")
tier3.add_command(create_dataset, "create-eval-dataset")
tier3.add_command(init_custom_grader, "init-custom-grader")
tier3.add_command(init_harbor_task, "init-harbor-task")
tier3.add_command(compare, "compare")
tier3.add_command(view, "view")
tier3.add_command(doctor, "doctor")


if __name__ == "__main__":
    cli()
