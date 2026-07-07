# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Harbor multi-agent evaluation mode.

Converts skill eval datasets to Harbor tasks, runs them against multiple
agents (Claude Code, Codex, Cursor, etc.) in isolated containers, grades
trajectories with SkillEvaluator metrics, and collects results.

Harbor mode is self-contained.

Environment modes are Harbor's built-in environment types. Docker is the
default, with cloud environments enabled through the corresponding Harbor
extra and provider credentials.
"""

from skillevaluator.tier3_environments import (
    DEFAULT_ENV_MODE,
    ENV_MODE_LOCAL,
    HARBOR_ENV_MODES,
    HARBOR_ENVIRONMENTS,
    HARBOR_NATIVE_ENV_MODES,
)

# Local mode plugs into Harbor as a custom environment/agents via import paths.
LOCAL_ENV_IMPORT_PATH = "skillevaluator.tier3.harbor.local_environment:SkillEvaluatorLocalEnvironment"
LOCAL_HARBOR_AGENTS = frozenset({"claude-code", "codex", "opencode"})
LOCAL_AGENT_IMPORT_PATHS = {
    "claude-code": "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorLocalClaudeCode",
    "codex": "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorLocalCodex",
    "opencode": "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorLocalOpenCode",
}

HARBOR_AGENTS_SUPPORTED = frozenset(
    {
        "claude-code",
        "cursor-cli",
        "openhands",
        "mini-swe-agent",
        "codex",
        "opencode",
        "cline-cli",
        "terminus-2",
    }
)

HARBOR_AGENTS_EXPERIMENTAL = frozenset(
    {
        "aider",
        "gemini-cli",
        "goose",
        "qwen-coder",
        "terminus",
        "oracle",
        "nop",
    }
)

HARBOR_AGENTS = HARBOR_AGENTS_SUPPORTED | HARBOR_AGENTS_EXPERIMENTAL

__all__ = [
    "DEFAULT_ENV_MODE",
    "ENV_MODE_LOCAL",
    "HARBOR_AGENTS",
    "HARBOR_AGENTS_EXPERIMENTAL",
    "HARBOR_AGENTS_SUPPORTED",
    "HARBOR_ENVIRONMENTS",
    "HARBOR_ENV_MODES",
    "HARBOR_NATIVE_ENV_MODES",
    "LOCAL_AGENT_IMPORT_PATHS",
    "LOCAL_ENV_IMPORT_PATH",
    "LOCAL_HARBOR_AGENTS",
]
