# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor 0.24 environment contract: the W&B backend move and the new runtime-policy options.

Harbor 0.24 folded its ``wandb`` backend into ``cwsandbox`` (``auth=wandb``)
and added ``stream`` and ``enable_environment_dir_upload`` to every
environment. SkillEvaluator must not offer a mode Harbor no longer has, must
keep W&B sandboxes usable through ``cwsandbox``, and must keep the new options
out of the operator ``--ek`` surface because they change how the sandbox runs.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

pytest.importorskip("harbor")

from skillevaluator.cli import cli
from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.harbor import runner
from skillevaluator.tier3_environments import HARBOR_ENV_MODES, HARBOR_PINNED_VERSION


def _provider() -> ProviderConfig:
    return ProviderConfig(
        provider="openai",
        model="gpt-test",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        litellm_model="openai/gpt-test",
    )


def test_wandb_mode_is_gone_and_the_cli_rejects_it() -> None:
    from harbor.models.environment_type import EnvironmentType

    assert "wandb" not in {environment.value for environment in EnvironmentType}
    assert "wandb" not in HARBOR_ENV_MODES

    result = CliRunner().invoke(cli, ["doctor", "--env-mode", "wandb"])

    assert result.exit_code != 0
    assert "wandb" in result.output


def test_wandb_sandboxes_run_as_cwsandbox_with_wandb_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    command = runner.build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="codex",
        job_name="wandb-auth",
        env_mode="cwsandbox",
        environment_kwargs={"auth": "wandb"},
    )

    assert command[command.index("--env") + 1] == "cwsandbox"
    assert 'auth="wandb"' in [command[index + 1] for index, value in enumerate(command) if value == "--ek"]
    with pytest.raises(ValueError, match='auth must be "wandb"'):
        runner.build_harbor_run_command(
            dataset_path="/tmp/dataset",
            agent="codex",
            job_name="wandb-auth",
            env_mode="cwsandbox",
            environment_kwargs={"auth": "sk-not-a-strategy"},
        )

    monkeypatch.setattr(
        runner.os,
        "environ",
        {"PATH": "/usr/bin", "HOME": "/home/test", "WANDB_API_KEY": "wandb-value", "E2B_API_KEY": "other"},
    )
    provider = _provider()
    environment = runner._harbor_subprocess_environment(
        env_mode="cwsandbox",
        provider=provider,
        configured_runtime_env={},
        provider_env=runner._provider_environment(provider),
    )

    assert environment["WANDB_API_KEY"] == "wandb-value"
    assert "E2B_API_KEY" not in environment


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
    with pytest.raises(ValueError, match=rf"Harbor {HARBOR_PINNED_VERSION.replace('.', '[.]')} environment 'e2b'"):
        runner.build_harbor_run_command(
            dataset_path="/tmp/dataset",
            agent="codex",
            job_name="unknown-kwarg",
            env_mode="e2b",
            environment_kwargs={"modal_sandbox_v2": True},
        )
