# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor 0.24 environment contract: the W&B alias and the new runtime-policy options.

Harbor 0.24 folded its ``wandb`` backend into ``cwsandbox`` (``auth=wandb``)
and added ``stream`` and ``enable_environment_dir_upload`` to every
environment. SkillEvaluator keeps ``--env-mode wandb`` as an alias that runs
Harbor's ``cwsandbox`` with W&B auth, never lets an operator choose ``auth``,
forwards the W&B credentials only to that alias, and keeps the new options out
of the operator ``--ek`` surface because they change how the sandbox runs.
``tests/test_harbor_environment_contract.py`` covers the alias command line
and its prerequisites.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

pytest.importorskip("harbor")

from skillevaluator.cli import cli
from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3 import commands as tier3_commands
from skillevaluator.tier3.harbor import runner
from skillevaluator.tier3_environments import HARBOR_ENV_MODES, HARBOR_VERSION, harbor_environment_type


def _provider() -> ProviderConfig:
    return ProviderConfig(
        provider="openai",
        model="gpt-test",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        litellm_model="openai/gpt-test",
    )


def test_wandb_is_a_skillevaluator_alias_for_harbor_cwsandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    from harbor.models.environment_type import EnvironmentType

    harbor_types = {environment.value for environment in EnvironmentType}
    assert "wandb" not in harbor_types
    assert "wandb" in HARBOR_ENV_MODES
    assert harbor_environment_type("wandb") == "cwsandbox"
    assert "cwsandbox" in harbor_types

    seen_env_modes: list[object] = []

    def fake_doctor(**kwargs: object) -> int:
        seen_env_modes.append(kwargs["env_mode"])
        return 0

    monkeypatch.setattr(tier3_commands, "doctor", fake_doctor)
    result = CliRunner().invoke(cli, ["doctor", "--env-mode", "wandb"])

    assert result.exit_code == 0, result.output
    assert seen_env_modes == ["wandb"]


def test_cwsandbox_rejects_an_operator_auth_strategy_without_echoing_it() -> None:
    with pytest.raises(ValueError, match="auth") as excinfo:
        runner.build_harbor_run_command(
            dataset_path="/tmp/dataset",
            agent="codex",
            job_name="cwsandbox-auth",
            env_mode="cwsandbox",
            environment_kwargs={"auth": "sk-not-a-strategy"},
        )

    assert "sk-not-a-strategy" not in str(excinfo.value)


def test_wandb_credentials_reach_only_the_wandb_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner.os,
        "environ",
        {"PATH": "/usr/bin", "HOME": "/home/test", "WANDB_API_KEY": "wandb-value", "E2B_API_KEY": "other"},
    )
    provider = _provider()

    def harbor_environment(env_mode: str) -> dict[str, str]:
        return runner._harbor_subprocess_environment(
            env_mode=env_mode,
            provider=provider,
            configured_runtime_env={},
            provider_env=runner._provider_environment(provider),
        )

    wandb_environment = harbor_environment("wandb")
    assert wandb_environment["WANDB_API_KEY"] == "wandb-value"
    assert "E2B_API_KEY" not in wandb_environment
    assert "WANDB_API_KEY" not in harbor_environment("cwsandbox")


@pytest.mark.parametrize("name", ["stream", "enable_environment_dir_upload"])
@pytest.mark.parametrize("env_mode", ["e2b", "daytona", "modal"])
def test_new_harbor_runtime_options_are_reserved(env_mode: str, name: str) -> None:
    with pytest.raises(ValueError, match=rf"reserved for Harbor runtime policy: {name}"):
        runner.build_harbor_run_command(
            dataset_path="/tmp/dataset",
            agent="codex",
            job_name="reserved-kwarg",
            env_mode=env_mode,
            environment_kwargs={name: True},
        )


def test_unknown_kwarg_message_names_the_pinned_harbor() -> None:
    with pytest.raises(ValueError, match=rf"Harbor {HARBOR_VERSION.replace('.', '[.]')} environment 'e2b'"):
        runner.build_harbor_run_command(
            dataset_path="/tmp/dataset",
            agent="codex",
            job_name="unknown-kwarg",
            env_mode="e2b",
            environment_kwargs={"modal_sandbox_v2": True},
        )
