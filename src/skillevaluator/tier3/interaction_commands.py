# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command surface for explicit skill-interaction experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from skillevaluator.cli_help import RichCommand
from skillevaluator.tier3_environments import all_env_modes


@click.command("interact", cls=RichCommand, context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("skills", nargs=-1, required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--eval-source",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Skill directory containing the single reviewed evals/ source used for every configuration.",
)
@click.option(
    "-a", "--agents", default=None, help="Comma-separated Harbor agents; defaults to the provider-native agent."
)
@click.option("--model", default=None, help="Global agent model override.")
@click.option("--env-mode", default="docker", show_default=True, type=click.Choice(all_env_modes()))
@click.option("--environment-kwarg", "--ek", multiple=True, help="Harbor backend kwarg KEY=VALUE; never pass secrets.")
@click.option(
    "--n-attempts",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Fixed attempts per case in every configuration; early stopping is disabled.",
)
@click.option("--n-concurrent", type=click.IntRange(min=1), default=None)
@click.option(
    "--metric",
    type=click.Choice(["goal_accuracy", "accuracy", "overall"]),
    default="goal_accuracy",
    show_default=True,
    help="Task outcome to compare; overall requires custom_only grading.",
)
@click.option(
    "--results-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output root outside the inputs; defaults to SKILLEVALUATOR_RESULTS_DIR or interaction-results beside --eval-source.",
)
@click.option("--harbor-keep-jobs", is_flag=True, help="Keep raw Harbor jobs as well as the collected evidence.")
@click.option("--json", "as_json", is_flag=True, help="Print the complete interaction report as JSON.")
def interact_command(
    skills: tuple[Path, ...],
    eval_source: Path,
    agents: str | None,
    model: str | None,
    env_mode: str,
    environment_kwarg: tuple[str, ...],
    n_attempts: int,
    n_concurrent: int | None,
    metric: str,
    results_dir: Path | None,
    harbor_keep_jobs: bool,
    as_json: bool,
) -> None:
    """Measure interaction among at least two skills on one fixed task suite.

    Run no skills, each skill alone, and all skills together. Positive means
    more than additive improvement; negative means less; zero means additive.
    """
    import json
    from contextlib import redirect_stdout

    from skillevaluator.evaluation import EvaluationOptions
    from skillevaluator.tier3.harbor.progress import create_progress_reporter
    from skillevaluator.tier3.interactions import run_interaction

    try:
        with redirect_stdout(click.get_text_stream("stderr")):
            report = run_interaction(
                skills,
                eval_source=eval_source,
                options=EvaluationOptions(
                    skill_path=eval_source,
                    agents=agents,
                    model=model,
                    env_mode=env_mode,
                    environment_kwarg=environment_kwarg,
                    n_attempts=n_attempts,
                    n_concurrent=n_concurrent,
                    results_dir=results_dir,
                    harbor_keep_jobs=harbor_keep_jobs,
                ),
                metric=metric,
                on_configuration=lambda name: click.echo(f"Tier 3 interaction: running {name}", err=True),
                progress_reporter=create_progress_reporter("auto", stream=click.get_text_stream("stderr")),
            )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    except OSError as exc:
        raise click.ClickException(str(exc)) from exc
    if as_json:
        click.echo(json.dumps(report, indent=2, allow_nan=False))
    else:
        _print_report(report)
    if report["status"] != "complete":
        raise click.exceptions.Exit(1)


def _print_report(report: dict[str, Any]) -> None:
    click.echo(f"Interaction evaluation: {report['status']} ({report['metric']})")
    for reason in dict.fromkeys(report.get("reasons", [])):
        click.echo(f"  {reason}")
    names = [skill["name"] for skill in report["skills"]]
    for agent, score in report.get("agents", {}).items():
        click.echo(f"{agent}: {score['status']}")
        if score["status"] != "complete":
            for reason in dict.fromkeys(score.get("reasons", [])):
                click.echo(f"  {reason}")
            continue
        click.echo(f"  Baseline: {score['baseline']:.4f}")
        for name, value in zip(names, score["individuals"], strict=True):
            click.echo(f"  {name}: {value:.4f}")
        click.echo(f"  Combined: {score['combined']:.4f}; combined lift: {score['combined_lift']:+.4f}")
        click.echo(f"  Interaction: {score['interaction']:+.4f} ({score['classification']})")
    click.echo("Negative interaction means less than additive gain; combined performance may still exceed baseline.")
    click.echo(f"Evidence: {Path(report['run_dir']) / 'interaction.json'}")
