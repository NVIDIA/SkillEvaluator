# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The loose-parts arm of a Tier 3 plugin run, on native Claude Code (v1).

The Integration lift (glue) is *full plugin minus loose parts*. Loose parts are
the same skills, rules, MCP servers, subagents, commands and hooks the full
plugin arm stages, put where Claude Code loads them standalone, with no plugin
around them. Only what cannot load standalone is left out, and it is listed as
glue with the reason (``dropped_as_glue``).

Where each piece goes (``$CLAUDE_CONFIG_DIR`` is the run's Claude Code config
dir, which Harbor sets; ``/skilleval`` is the native bundle in the image):

* member skills: the task skills projection (Harbor copies them to
  ``$CLAUDE_CONFIG_DIR/skills/``), exactly as the skills-only arm;
* rules: user rules ``$CLAUDE_CONFIG_DIR/rules/skilleval-<plugin>/<rule>.md``,
  written by the plugin arm's own rule writer, so the bytes match;
* subagents and commands: ``$CLAUDE_CONFIG_DIR/agents/`` and ``commands/``,
  byte-identical copies;
* hooks: ``$CLAUDE_CONFIG_DIR/settings.json`` (``{"hooks": ...}``), the plugin
  arm's census-wrapped hooks with ``${CLAUDE_PLUGIN_ROOT}`` replaced by
  :data:`LOOSE_ROOT` (and ``${CLAUDE_PLUGIN_DATA}`` by :data:`LOOSE_DATA`);
* MCP servers: :data:`LOOSE_MCP_CONFIG`, passed with ``--mcp-config``: the
  plugin arm's ``.mcp.json`` entries (command, args, env) with the same
  substitution;
* every plugin file a hook or MCP server references through
  ``${CLAUDE_PLUGIN_ROOT}``: a byte- and mode-identical copy under
  :data:`LOOSE_ROOT` at the same relative path.

Everything reaches the container through the native staging path
(:func:`~skillevaluator.tier3.harbor.native_staging.stage_native_bundle` and
``/skilleval/native/setup.sh``, run right before the agent starts by the
Harbor wrappers in :mod:`skillevaluator.tier3.harbor.loose_agents`).

Content hashes prove the two arms carry the same capability: every piece is
hashed as each arm stages it (staged into a private directory by the same
code the runner uses), after a documented normalization (the loose root
written back as ``${CLAUDE_PLUGIN_ROOT}``; a file the plugin arm rewrites only
to census-wrap its frontmatter hooks is hashed from its source).

Load is verified only by Claude Code's own ``system/init`` report
(:func:`loose_parts_load_evidence`), with one documented workaround: a
successful call of an MCP server's tool proves a server that init reported as
``pending`` (or not at all). Rules and hooks are not in that report.

This module is host-side and imports no Harbor code.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import shlex
import stat
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from skillevaluator.constants import CONTENT_DEDUP_MAX_FILE_BYTES
from skillevaluator.plugin_paths import PluginRootReader
from skillevaluator.tier3.harbor.stats import ARM_LOOSE_PARTS, ARM_SUM_OF_PARTS, ARM_WITH, ARM_WITHOUT
from skillevaluator.tier3.plugin_native import (
    BUNDLE_DIRNAME,
    CONTAINER_ROOT,
    HOOK_CENSUS_SCRIPT,
    NATIVE_ROOT,
    CensusCheck,
    ClaudeCodeAdapter,
    HarnessAdapter,
    NativeBundle,
    NativePluginSource,
    PluginLoadError,
    _claude_rule_file_name,
    _claude_user_rule,
    _mcp_entry_claude,
    _redacted,
    _staged_handler_needle,
    _staged_hook_events,
    is_frontmatter_hook_source,
    native_refusal,
    plugin_slug,
    render_setup_script,
    safe_name,
    shell_path,
    wrap_hook_sources,
)

#: The arm key in results, lift JSON, and plugin signals (the statistics' own key), and its report label.
LOOSE_PARTS_ARM = ARM_LOOSE_PARTS
LOOSE_PARTS_LABEL = "Loose parts"
#: The one harness v1 measures.
LOOSE_AGENT = "claude-code"
LOOSE_ADAPTER_ID = "claude-code-loose-parts"
#: The census harness kind; the plugin-arm ``claude-code-init`` check must never read this arm.
LOOSE_HARNESS_KIND = "claude-code-loose-init"
#: Bundle-relative directory of the loose parts, and where it lands in the container.
LOOSE_BUNDLE_DIR = "native/claude-code/loose"
LOOSE_DIR = f"{CONTAINER_ROOT}/{LOOSE_BUNDLE_DIR}"
#: Stands in for ``${CLAUDE_PLUGIN_ROOT}``: referenced plugin files keep their relative path under it.
LOOSE_ROOT = f"{LOOSE_DIR}/root"
#: Stands in for ``${CLAUDE_PLUGIN_DATA}`` (created empty by setup, like a fresh plugin data dir).
LOOSE_DATA = "/tmp/skilleval-loose-data"
#: The plain MCP config the launch passes with ``--mcp-config``.
LOOSE_MCP_CONFIG = f"{LOOSE_DIR}/mcp.json"

#: Arm aliases accepted by the hashing and evidence functions -> the result arm key.
ARM_ALIASES: dict[str, str] = {
    "none": ARM_WITHOUT,
    ARM_WITHOUT: ARM_WITHOUT,
    "skills": ARM_SUM_OF_PARTS,
    ARM_SUM_OF_PARTS: ARM_SUM_OF_PARTS,
    "loose": LOOSE_PARTS_ARM,
    LOOSE_PARTS_ARM: LOOSE_PARTS_ARM,
    "full": ARM_WITH,
    ARM_WITH: ARM_WITH,
}
#: The staged arms whose content hashes are compared.
COMPARED_ARMS = (ARM_WITH, LOOSE_PARTS_ARM, ARM_SUM_OF_PARTS)

NOT_SUPPORTED_WRAPPER = (
    "not supported: use native (--plugin-load native). In wrapper mode the plugin arm is SkillEvaluator's own "
    "generated wrapper SKILL.md, so full plugin minus loose parts would measure that wrapper, not the plugin's glue"
)
NOTHING_LOOSE = (
    "loose parts equal the skills alone: the plugin has no rule, runnable MCP server, subagent, command, or hook "
    "that Claude Code can load standalone, so a loose-parts arm would repeat the skills-only arm"
)

# Init-verifiable piece kinds; rules, hooks, and scripts are not in Claude Code's startup report.
_INIT_KINDS = frozenset({"skill", "agent", "command", "mcp"})
#: The hook piece of every staged hook group without a command handler (and so without a hook id).
UNATTRIBUTED_HOOKS = "(groups without a command handler)"
_EVIDENCE_INIT = "system/init"
_EVIDENCE_CALL = "mcp-call"
_EVIDENCE_NONE = "unverifiable"
_MAX_REFERENCED_FILES = 2048
_PLUGIN_BUNDLE = f"{BUNDLE_DIRNAME}/{ClaudeCodeAdapter.bundle_dir}"
_LOOSE_BUNDLE = f"{BUNDLE_DIRNAME}/{LOOSE_BUNDLE_DIR}"
_RULES_BUNDLE = f"{BUNDLE_DIRNAME}/native/claude-code/rules"
_ROOT_VAR = "${CLAUDE_PLUGIN_ROOT}"
_DATA_VAR = "${CLAUDE_PLUGIN_DATA}"
# Only the braced form is replaced: Claude Code expands ``${VAR}`` in MCP fields, and a shell expands the bare
# ``$CLAUDE_PLUGIN_ROOT`` from the environment, which the loose launch sets (so both arms read it the same way).
_ROOT_VAR_RE = re.compile(r"\$\{CLAUDE_PLUGIN_ROOT\}")
_DATA_VAR_RE = re.compile(r"\$\{CLAUDE_PLUGIN_DATA\}")
# A root reference and the plugin path after it (``${CLAUDE_PLUGIN_ROOT}/x.sh``, ``"${CLAUDE_PLUGIN_ROOT}"/x.sh``).
# ``:`` and ``,`` end a path, so ``PYTHONPATH=${CLAUDE_PLUGIN_ROOT}/lib:${CLAUDE_PLUGIN_ROOT}/src`` names two.
_ROOT_PATH_RE = re.compile(
    r"(?:\$\{CLAUDE_PLUGIN_ROOT\}|\$CLAUDE_PLUGIN_ROOT(?![A-Za-z0-9_]))[\"']?(/[^\s\"'`;|&<>(){}$\\:,]*)?"
)
_HOOK_ID_SOURCE_RE = re.compile(r"^(?P<source>.+)#[^#]*\[\d+\]\.hooks\[\d+\]$")
_SERVER_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")

_GLUE_REASONS = {
    "output_style": (
        "output styles change the whole session's presentation; v1 counts a plugin output style as glue, "
        "not a standalone capability"
    ),
    "lsp": "LSP servers load only from a plugin's .lsp.json; Claude Code has no standalone user-level LSP config",
    "settings": "plugin settings.json keys apply only while the plugin is enabled (they configure the plugin session)",
    "bin": "bin/ executables join the Bash PATH only while the plugin is enabled",
    "manifest": (
        "the plugin manifest: the plugin identity, its <plugin>: namespacing of skills, agents, commands and MCP "
        "servers, and the plugin listing; loose parts have none of these by definition"
    ),
    "userConfig": "userConfig values are plugin install-time settings that exist only for an installed plugin",
}


class LoosePartsUnsupported(PluginLoadError):
    """The loose-parts arm cannot run for this agent or load mode (the message starts with ``not supported:``)."""


def not_supported_reason(agent: str) -> str:
    """Why an agent other than Claude Code has no loose-parts arm in v1."""
    return (
        f"not supported: the loose-parts arm runs on native Claude Code only in v1; {agent} is not measured "
        "(Codex, OpenCode and Hermes: not measured)"
    )


def loose_parts_launch_env() -> dict[str, str]:
    """The plugin-root variables the loose launch exports, so scripts reading them find the loose root."""
    return {"CLAUDE_PLUGIN_ROOT": LOOSE_ROOT, "CLAUDE_PLUGIN_DATA": LOOSE_DATA}


# --------------------------------------------------------------------------- #
# Public records                                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LoosePiece:
    """One loose part: ``kind:name`` id, the plugin-relative file it came from, and its loose-arm sha256."""

    id: str
    kind: str
    source: str | None
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "source": self.source, "sha256": self.sha256}


@dataclass(frozen=True)
class GluePiece:
    """A plugin part that is not a loose part, with the reason."""

    id: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "reason": self.reason}


@dataclass(frozen=True)
class _Placement:
    """Where one piece is staged in one arm, relative to the task's ``environment/`` dir.

    ``key`` picks the piece out of a JSON file (an MCP server name or a hook
    source name). ``overrides`` maps a staged file to the source file hashed
    instead (a census rewrite), and ``exclude`` lists staged files a tree hash
    leaves out (files only the plugin arm generates).
    """

    id: str
    kind: str
    path: str
    source: str | None = None
    key: str | None = None
    overrides: tuple[tuple[str, Path], ...] = ()
    exclude: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.id.split(":", 1)[1]


@dataclass(frozen=True)
class LooseParts:
    """The loose-parts arm for one plugin and agent (see the module docstring).

    ``bundle`` is the native bundle :func:`stage_native_bundle` consumes (``None``
    with ``error``). ``error`` starts with ``not supported:`` for a harness or
    load mode v1 does not measure, and is :data:`NOTHING_LOOSE` for a plugin
    whose loose parts equal its skills.
    """

    agent: str
    bundle: NativeBundle | None
    dropped_as_glue: tuple[GluePiece, ...] = ()
    not_staged: tuple[GluePiece, ...] = ()
    rules: tuple[str, ...] = ()
    mcp_servers: tuple[str, ...] = ()
    error: str | None = None
    source: NativePluginSource | None = field(default=None, repr=False, compare=False)
    _loose: tuple[_Placement, ...] = field(default=(), repr=False, compare=False)
    _full: tuple[_Placement, ...] = field(default=(), repr=False, compare=False)
    _normalized: tuple[tuple[str, str], ...] = field(default=(), repr=False, compare=False)

    # -- staging -------------------------------------------------------------- #
    def task_staging(self) -> Any:
        """The :class:`~skillevaluator.tier3.harbor.native_staging.NativeTaskStaging` for the loose arm's task."""
        from skillevaluator.tier3.harbor.native_staging import NativeTaskStaging

        if self.error is not None or self.bundle is None or self.source is None:
            raise _unsupported_or_load_error(self.error or NOTHING_LOOSE)
        adapter = LoosePartsClaudeCodeAdapter()
        refusal = native_refusal(adapter, self.source)
        if refusal is not None:
            raise PluginLoadError(refusal)
        return NativeTaskStaging(agent=self.agent, adapter=adapter, source=self.source, bundle=self.bundle)

    def stage_arm(self, arm: str, env_dir: Path, *, excluded_roots: Sequence[Path] = ()) -> None:
        """Stage one arm's plugin content into ``env_dir`` (a task's ``environment/``) with the runner's code.

        ``without_skill`` stages nothing, ``sum_of_parts`` the member skills
        (``skills/``), ``loose_parts`` the member skills and the loose bundle,
        and ``with_skill`` the native Claude Code plugin bundle.
        """
        from skillevaluator.tier3.harbor.native_staging import build_native_task_staging, stage_native_bundle

        arm = _arm_key(arm)
        source = self._require_source()
        excluded = (*excluded_roots, *source.excluded_paths)
        env_dir.mkdir(parents=True, exist_ok=True)
        if arm in {ARM_SUM_OF_PARTS, LOOSE_PARTS_ARM}:
            _stage_member_skills(env_dir / "skills", source.member_skills, excluded)
        if arm == LOOSE_PARTS_ARM:
            stage_native_bundle(env_dir, self.task_staging(), excluded_roots=excluded)
        elif arm == ARM_WITH:
            staging = build_native_task_staging(LOOSE_AGENT, ClaudeCodeAdapter(), source)
            stage_native_bundle(env_dir, staging, excluded_roots=excluded)

    # -- hashing -------------------------------------------------------------- #
    def hash_staged_arm(self, arm: str, env_dir: Path) -> dict[str, str]:
        """``{piece id: sha256}`` of every piece one arm staged into ``env_dir`` (an absent piece is left out)."""
        cache: dict[str, Any] = {}
        hashes: dict[str, str] = {}
        for placement in self._placements(_arm_key(arm)):
            digest = _piece_digest(env_dir, placement, cache)
            if digest is not None:
                hashes[placement.id] = digest
        return dict(sorted(hashes.items()))

    def content_hashes_for_arm(self, arm: str, *, excluded_roots: Sequence[Path] = ()) -> dict[str, str]:
        """Stage ``arm`` (``none``/``skills``/``loose``/``full`` or the arm key) privately and hash its pieces."""
        if self.error is not None:
            return {}
        with tempfile.TemporaryDirectory(prefix="skilleval-loose-hash-") as scratch:
            env_dir = Path(scratch) / "environment"
            self.stage_arm(arm, env_dir, excluded_roots=excluded_roots)
            return self.hash_staged_arm(arm, env_dir)

    def content_report(self, *, excluded_roots: Sequence[Path] = ()) -> dict[str, Any]:
        """``content_hashes`` of the full, loose, and skills-only arms, ``equal_across_arms`` and ``mismatches``."""
        if self.error is not None:
            return {"content_hashes": {}, "equal_across_arms": None, "mismatches": []}
        report = compare_content_hashes(
            {arm: self.content_hashes_for_arm(arm, excluded_roots=excluded_roots) for arm in COMPARED_ARMS}
        )
        if self._normalized:
            report["normalized"] = dict(self._normalized)
        return report

    @cached_property
    def pieces(self) -> tuple[LoosePiece, ...]:
        """Every staged loose part with its loose-arm sha256 (staged privately once, then cached)."""
        if self.error is not None:
            return ()
        return self._pieces(self.content_hashes_for_arm(LOOSE_PARTS_ARM))

    def _pieces(self, hashes: Mapping[str, str]) -> tuple[LoosePiece, ...]:
        return tuple(
            LoosePiece(placement.id, placement.kind, placement.source, hashes.get(placement.id, ""))
            for placement in sorted(self._loose, key=lambda item: item.id)
        )

    # -- records -------------------------------------------------------------- #
    def census_plan(self) -> dict[str, Any]:
        """What the collector needs to read this arm's load census (``mode``, declared parts, hook ids, harness)."""
        if self.bundle is None:
            return {}
        return {"mode": "native", "arm": LOOSE_PARTS_ARM, **self.bundle.census_plan()}

    def info(
        self,
        *,
        content: Mapping[str, Any] | None = None,
        load_evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The ``loose_parts`` JSON block: what the arm staged, the content hashes, glue, and load evidence.

        ``content`` defaults to :meth:`content_report` (it stages the three arms
        privately); ``load_evidence`` is the per-case/per-arm evidence the
        collector gathered (empty until it runs).
        """
        report = dict(content) if content is not None else self.content_report()
        loose_hashes = report.get("content_hashes", {}).get(LOOSE_PARTS_ARM)
        # The piece hashes come from the same staging as the report, so the block never disagrees with itself.
        pieces = self._pieces(loose_hashes) if isinstance(loose_hashes, Mapping) else self.pieces
        block: dict[str, Any] = {
            "arm": LOOSE_PARTS_ARM,
            "label": LOOSE_PARTS_LABEL,
            "agent": self.agent,
            "adapter": LOOSE_ADAPTER_ID,
            "error": self.error,
            "loose_dir": LOOSE_DIR,
            "loose_root": LOOSE_ROOT,
            "mcp_config": LOOSE_MCP_CONFIG,
            "pieces": [piece.to_dict() for piece in pieces],
            "rules": list(self.rules),
            "mcp_servers": list(self.mcp_servers),
            "content_hashes": report.get("content_hashes", {}),
            "equal_across_arms": report.get("equal_across_arms"),
            "mismatches": list(report.get("mismatches", [])),
            "dropped_as_glue": [piece.to_dict() for piece in self.dropped_as_glue],
            "not_staged": [piece.to_dict() for piece in self.not_staged],
            "load_evidence": dict(load_evidence or {}),
        }
        if report.get("normalized"):
            block["normalized"] = dict(report["normalized"])
        return block

    # -- internals ------------------------------------------------------------ #
    def _require_source(self) -> NativePluginSource:
        if self.source is None or self.error is not None:
            raise _unsupported_or_load_error(self.error or NOTHING_LOOSE)
        return self.source

    def _placements(self, arm: str) -> tuple[_Placement, ...]:
        if arm == LOOSE_PARTS_ARM:
            return self._loose
        if arm == ARM_WITH:
            return self._full
        if arm == ARM_SUM_OF_PARTS:
            return tuple(placement for placement in self._loose if placement.kind == "skill")
        return ()


class LoosePartsClaudeCodeAdapter(HarnessAdapter):
    """Claude Code with the plugin's parts staged as plain user-level pieces: no plugin, no manifest."""

    agent = LOOSE_AGENT
    adapter_id = LOOSE_ADAPTER_ID
    description = (
        "stages the plugin's skills, rules, MCP servers, subagents, commands and hooks as plain Claude Code "
        "user-level pieces under $CLAUDE_CONFIG_DIR and --mcp-config; no plugin, no manifest"
    )
    components: ClassVar[Mapping[str, str]] = {
        "skill": "native",
        "rule": "native",
        "mcp": "native",
        "hook": "native",
        "agent": "native",
        "command": "native",
    }
    stage_wrapper_skill = False
    #: Member skills go through the task skills projection, exactly as in the skills-only arm.
    stage_member_skills = True
    plugin_mcp_via_task = False
    #: The plugin files that hooks and MCP servers reference are copied, so plugin-file servers start.
    copies_plugin_tree = True

    def build(self, source: NativePluginSource) -> NativeBundle:
        parts = build_loose_parts(source)
        if parts.error is not None or parts.bundle is None:
            raise _unsupported_or_load_error(parts.error or NOTHING_LOOSE)
        return parts.bundle


# --------------------------------------------------------------------------- #
# Building the loose arm                                                       #
# --------------------------------------------------------------------------- #
def build_loose_parts(
    source: NativePluginSource | None,
    agent: str = LOOSE_AGENT,
    *,
    plugin_load: str = "native",
) -> LooseParts:
    """Plan the loose-parts arm for ``agent`` from the native plugin snapshot.

    Never raises for an unsupported agent or load mode, or a plugin with
    nothing loose: ``error`` says why. Raises ``ValueError`` for content that
    must never be staged (a referenced plugin file that is a link or special
    file, two parts with one destination, an inline literal secret).
    """
    if plugin_load == "wrapper" or source is None:
        return LooseParts(agent=agent, bundle=None, error=NOT_SUPPORTED_WRAPPER, source=source)
    if agent != LOOSE_AGENT:
        return LooseParts(agent=agent, bundle=None, error=not_supported_reason(agent), source=source)
    refusal = native_refusal(LoosePartsClaudeCodeAdapter(), source)
    if refusal is not None:
        return LooseParts(agent=agent, bundle=None, error=refusal, source=source)
    return _Builder(source).build()


def loose_parts_error(
    source: NativePluginSource | None, agent: str = LOOSE_AGENT, *, plugin_load: str = "native"
) -> str | None:
    """Why the loose-parts arm cannot run for this plugin snapshot and agent, or ``None``."""
    try:
        return build_loose_parts(source, agent, plugin_load=plugin_load).error
    except (ValueError, OSError) as exc:
        return f"loose parts cannot be staged: {exc}"


def loose_parts_task_staging(agent: str, source: NativePluginSource | None, *, plugin_load: str = "native") -> Any:
    """The runner's seam: the loose arm's native task staging for ``agent``.

    Returns the :class:`~skillevaluator.tier3.harbor.native_staging.NativeTaskStaging`
    that ``generate_harbor_tasks(native_plugin=...)`` and ``stage_native_bundle``
    consume. Raises :class:`LoosePartsUnsupported` (``not supported: ...``) for
    another harness or wrapper mode, and :class:`PluginLoadError` for a
    permission-bypass refusal or a plugin with nothing loose.
    """
    if plugin_load == "wrapper" or source is None:
        raise LoosePartsUnsupported(NOT_SUPPORTED_WRAPPER)
    if agent != LOOSE_AGENT:
        raise LoosePartsUnsupported(not_supported_reason(agent))
    refusal = native_refusal(LoosePartsClaudeCodeAdapter(), source)
    if refusal is not None:
        raise PluginLoadError(refusal)
    return build_loose_parts(source, agent, plugin_load=plugin_load).task_staging()


_LOOSE_AGENTS = "skillevaluator.tier3.harbor.loose_agents"
#: (agent, base Harbor import path or ``None`` for the stock agent) -> loose-parts wrapper import path.
LOOSE_PARTS_AGENT_IMPORT_PATHS: dict[tuple[str, str | None], str] = {
    (LOOSE_AGENT, None): f"{_LOOSE_AGENTS}:LoosePartsClaudeCode",
    (
        LOOSE_AGENT,
        "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorNvidiaBuildClaudeCode",
    ): f"{_LOOSE_AGENTS}:LoosePartsNvidiaBuildClaudeCode",
}


def loose_parts_agent_import_path(agent: str, base_import_path: str | None) -> str:
    """The Harbor wrapper that runs the loose arm's setup before ``agent`` starts, or raise ``LoosePartsUnsupported``."""
    if agent != LOOSE_AGENT:
        raise LoosePartsUnsupported(not_supported_reason(agent))
    try:
        return LOOSE_PARTS_AGENT_IMPORT_PATHS[(agent, base_import_path)]
    except KeyError:
        raise LoosePartsUnsupported(
            f"not supported: the loose-parts arm has no Harbor wrapper for {agent} over "
            f"{base_import_path or 'the stock agent'}"
        ) from None


def _unsupported_or_load_error(message: str) -> PluginLoadError:
    return LoosePartsUnsupported(message) if message.startswith("not supported:") else PluginLoadError(message)


def _arm_key(arm: str) -> str:
    try:
        return ARM_ALIASES[arm]
    except KeyError:
        raise ValueError(f"unknown arm for loose-parts hashing: {arm!r}") from None


def _to_loose(value: Any) -> Any:
    """Replace ``${CLAUDE_PLUGIN_ROOT}`` and ``${CLAUDE_PLUGIN_DATA}`` with the loose directories, in every string."""
    if isinstance(value, str):
        value = _ROOT_VAR_RE.sub(lambda _match: LOOSE_ROOT, value)
        return _DATA_VAR_RE.sub(lambda _match: LOOSE_DATA, value)
    if isinstance(value, list):
        return [_to_loose(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_loose(item) for key, item in value.items()}
    return value


def _canonical_roots(value: Any) -> Any:
    """The hashing normalization: the loose directories written back as the variables they replaced."""
    if isinstance(value, str):
        return value.replace(LOOSE_ROOT, _ROOT_VAR).replace(LOOSE_DATA, _DATA_VAR)
    if isinstance(value, list):
        return [_canonical_roots(item) for item in value]
    if isinstance(value, dict):
        return {key: _canonical_roots(item) for key, item in value.items()}
    return value


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)


class _Builder:
    """Plans one loose-parts bundle; every write goes to ``bundle`` and every decision to a record."""

    def __init__(self, source: NativePluginSource) -> None:
        self.source = source
        self.adapter = ClaudeCodeAdapter()
        self.slug = self.adapter.plugin_name(source)
        self.reader = PluginRootReader(source.plugin_root)
        self.full_bundle = self.adapter.build(source)
        self.bundle = NativeBundle(
            agent=LOOSE_AGENT, adapter=LOOSE_ADAPTER_ID, components=LoosePartsClaudeCodeAdapter().component_modes()
        )
        self.checks: list[CensusCheck] = []
        self.not_loaded: list[tuple[str, str, str]] = []
        self.setup: list[str] = [': "${CLAUDE_CONFIG_DIR:=$HOME/.claude}"']
        self.loose: list[_Placement] = []
        self.full: list[_Placement] = []
        self.glue: list[GluePiece] = []
        self.not_staged: list[GluePiece] = []
        self.references: list[str] = []
        self.normalized: dict[str, str] = {}
        self.staged_skills = self.adapter.staged_skills(source)
        self.rewrites = self._frontmatter_rewrites()
        # Paths inside the plugin copy that are not plugin source files: the files the plugin arm generates and
        # the member skills it copies in from outside the plugin root. A tree hash of plugin files leaves them out.
        self.generated_in_plugin = tuple(
            sorted(
                rel.removeprefix(f"{ClaudeCodeAdapter.bundle_dir}/")
                for rel in (*self.full_bundle.generated, *(dest for _source, dest in self.full_bundle.trees))
                if rel.startswith(f"{ClaudeCodeAdapter.bundle_dir}/")
            )
        )

    # -- the build ------------------------------------------------------------ #
    def build(self) -> LooseParts:
        self._skills()
        rules = self._rules()
        servers = self._mcp()
        hooks = self._hooks()
        texts = self._texts()
        self._scripts()
        self._glue()
        if not (rules or servers or hooks or texts):
            return LooseParts(
                agent=LOOSE_AGENT,
                bundle=None,
                dropped_as_glue=tuple(sorted(self.glue, key=lambda piece: piece.id)),
                not_staged=tuple(sorted(self.not_staged, key=lambda piece: piece.id)),
                error=NOTHING_LOOSE,
                source=self.source,
            )
        # The launch always exports CLAUDE_PLUGIN_DATA, so its directory always exists (empty, as for a new plugin).
        self.setup.append(f"mkdir -p {LOOSE_DATA} 2>/dev/null || skilleval_note mkdir:{LOOSE_DATA}")
        self.bundle.declared = [check.declared() for check in self.checks]
        self.bundle.harness = {"kind": LOOSE_HARNESS_KIND, "plugin": self.slug}
        self.bundle.files.sort(key=lambda item: item[1])
        self.bundle.generated["native/setup.sh"] = render_setup_script(
            agent=LOOSE_AGENT, setup_lines=self.setup, checks=self.checks, not_loaded=self.not_loaded
        )
        return LooseParts(
            agent=LOOSE_AGENT,
            bundle=self.bundle,
            dropped_as_glue=tuple(sorted(self.glue, key=lambda piece: piece.id)),
            not_staged=tuple(sorted(self.not_staged, key=lambda piece: piece.id)),
            rules=tuple(sorted(rules)),
            mcp_servers=tuple(servers),
            source=self.source,
            _loose=tuple(sorted(self.loose, key=lambda item: item.id)),
            _full=tuple(sorted(self.full, key=lambda item: item.id)),
            _normalized=tuple(sorted(self.normalized.items())),
        )

    def _place(self, piece_id: str, kind: str, source: str | None, *, full: str, loose: str, **extra: Any) -> None:
        full_overrides = tuple(
            (staged, original)
            for staged, original in self.rewrites.items()
            if staged == full or staged.startswith(f"{full}/")
        )
        if full_overrides:
            self.normalized[piece_id] = (
                "the plugin arm census-wraps this file's frontmatter hooks; the source file is hashed for that arm"
            )
        self.full.append(_Placement(piece_id, kind, full, source, overrides=full_overrides, **extra))
        self.loose.append(_Placement(piece_id, kind, loose, source, **extra))

    def _frontmatter_rewrites(self) -> dict[str, Path]:
        """Staged plugin-arm file -> its source, for each file the plugin arm rewrites to census-wrap frontmatter hooks.

        The rewrites are the plugin bundle's own generated files (``stage_frontmatter_hooks`` output), found at
        the path the plugin arm stages each hook's file (a relocated member skill keeps its new directory).
        """
        root = self.source.plugin_root.resolve()
        relocated = {
            copied.resolve().relative_to(root).as_posix(): staged
            for _name, staged, copied in self.staged_skills
            if copied is not None and copied.resolve().is_relative_to(root)
        }
        rewrites: dict[str, Path] = {}
        for hook in self.source.hooks:
            if not is_frontmatter_hook_source(hook) or hook.rel is None:
                continue
            rel = PurePosixPath(hook.rel)
            staged = (PurePosixPath(relocated.get(rel.parent.as_posix(), rel.parent.as_posix())) / rel.name).as_posix()
            if f"{ClaudeCodeAdapter.bundle_dir}/{staged}" in self.full_bundle.generated:
                rewrites[f"{_PLUGIN_BUNDLE}/{staged}"] = self.source.plugin_root / hook.rel
        return rewrites

    # -- pieces --------------------------------------------------------------- #
    def _skills(self) -> None:
        full_dirs = {PurePosixPath(rel).name: rel for _name, rel, _copied in self.staged_skills}
        loose_names = set()
        for skill in self.source.member_skills:
            name = skill.name
            loose_names.add(name)
            self._note_references(_skill_manifest_text(skill))
            self.checks.append(
                CensusCheck(
                    "skill", name, "file", f"$CLAUDE_CONFIG_DIR/skills/{name}/SKILL.md", label="user skills listing"
                )
            )
            rel = full_dirs.get(name)
            source_rel = _plugin_rel(self.source.plugin_root, skill)
            if rel is None:
                self.loose.append(_Placement(f"skill:{name}", "skill", f"skills/{name}", source_rel))
                continue
            self._place(f"skill:{name}", "skill", source_rel, full=f"{_PLUGIN_BUNDLE}/{rel}", loose=f"skills/{name}")
        for name, rel in full_dirs.items():
            if name not in loose_names:
                # The plugin arm loads a skill that is not a member skill: a mismatch the hashes report.
                self.full.append(_Placement(f"skill:{name}", "skill", f"{_PLUGIN_BUNDLE}/{rel}", rel))

    def _rules(self) -> list[str]:
        """Stage the rules with the plugin arm's own writer, so path and bytes are the same in both arms."""
        not_loaded: list[tuple[str, str, str]] = []
        lines = self.adapter._stage_rules(self.bundle, self.source, self.checks, not_loaded)
        # The rule writer starts with the same CLAUDE_CONFIG_DIR default line this setup already has.
        self.setup.extend(line for line in lines if line not in self.setup)
        self.not_loaded.extend(not_loaded)
        staged: list[str] = []
        for name, content in self.source.rules:
            text, reason = _claude_user_rule(name, content)
            if text is None:
                self.not_staged.append(GluePiece(f"rule:{name}", f"not staged in any arm: {reason}"))
                continue
            path = f"{_RULES_BUNDLE}/{_claude_rule_file_name(name)}"
            self._place(f"rule:{name}", "rule", _rule_source(self.source, name), full=path, loose=path)
            staged.append(name)
        return staged

    def _mcp(self) -> list[str]:
        """Write the plain MCP config: the plugin arm's ``.mcp.json`` entries with the root variables replaced."""
        entries = {
            _redacted(str(server["name"])): _mcp_entry_claude(server, self.source.mcp_declared.get(str(server["name"])))
            for server in (*self.source.mcp_servers, *self.source.plugin_file_mcp_servers)
        }
        for entry in entries.values():
            self._note_references(entry)
        loose_entries = {name: _to_loose(entry) for name, entry in entries.items()}
        self.bundle.generated[f"{LOOSE_BUNDLE_DIR}/mcp.json"] = (
            json.dumps({"mcpServers": loose_entries}, indent=2) + "\n"
        )
        for name in entries:
            self.checks.append(
                CensusCheck(
                    "mcp", name, "contains", LOOSE_MCP_CONFIG, needle=json.dumps(name), label="--mcp-config listing"
                )
            )
            self._place(
                f"mcp:{name}",
                "mcp",
                ".mcp.json",
                full=f"{_PLUGIN_BUNDLE}/.mcp.json",
                loose=f"{_LOOSE_BUNDLE}/mcp.json",
                key=name,
            )
        return list(entries)

    def _hooks(self) -> list[str]:
        """Write the user ``settings.json``: the plugin arm's census-wrapped hooks with the root variables replaced."""
        wrapped = wrap_hook_sources(self.source.hooks, plugin_root=self.source.plugin_root)
        staged: list[str] = []
        for hook in self.source.hooks:
            if is_frontmatter_hook_source(hook):
                # Stays inside its skill or command file, staged as written; the files it names through
                # the plugin root are copied, and the launch exports CLAUDE_PLUGIN_ROOT for its shell.
                self._note_references(hook.config)
                continue
            identifiers = [identifier for owner, _event, identifier in wrapped.ids if owner == hook.name]
            dropped = [event for owner, event in wrapped.dropped if owner == hook.name]
            for event in dropped:
                self.not_staged.append(
                    GluePiece(
                        f"hook:{hook.name}#{event}",
                        f"not staged in any arm: the Cursor hook event '{event}' has no Claude Code event",
                    )
                )
            needle = (
                json.dumps(identifiers[0])[1:-1] if identifiers else _staged_handler_needle(_staged_hook_events(hook))
            )
            if needle is None:
                self.not_staged.append(
                    GluePiece(f"hook:{hook.name}", "not staged in any arm: no hook handler Claude Code loads")
                )
                continue
            self.checks.append(
                CensusCheck(
                    "hook",
                    hook.name,
                    "contains",
                    "$CLAUDE_CONFIG_DIR/settings.json",
                    needle=_to_loose(needle),
                    label="user settings hooks listing",
                )
            )
            staged.append(hook.name)
            if not identifiers:
                continue  # no command handler, so no hook id: hashed with the other unattributed groups below
            self.bundle.hook_ids[hook.name] = identifiers
            self._place(
                f"hook:{hook.name}",
                "hook",
                hook.rel,
                full=f"{_PLUGIN_BUNDLE}/hooks/hooks.json",
                loose=f"{_LOOSE_BUNDLE}/settings.json",
                key=hook.name,
            )
        if not staged:
            return staged
        events = wrapped.config["hooks"]
        if _hook_groups({"hooks": events}, ""):
            # Groups with only prompt, agent, or http handlers carry no hook id to name their source.
            self._place(
                f"hook:{UNATTRIBUTED_HOOKS}",
                "hook",
                None,
                full=f"{_PLUGIN_BUNDLE}/hooks/hooks.json",
                loose=f"{_LOOSE_BUNDLE}/settings.json",
                key="",
            )
        self._note_references(events)
        settings = {"hooks": _to_loose(events)}
        self.bundle.generated[f"{LOOSE_BUNDLE_DIR}/settings.json"] = json.dumps(settings, indent=2) + "\n"
        dest = shell_path("$CLAUDE_CONFIG_DIR/settings.json")
        source = shlex.quote(f"{NATIVE_ROOT}/claude-code/loose/settings.json")
        self.setup.append(
            f'if [ -e {dest} ] || [ -L {dest} ]; then skilleval_note "exists:$CLAUDE_CONFIG_DIR/settings.json"; '
            f"else skilleval_copy {source} {dest}; fi"
        )
        return staged

    def _texts(self) -> list[str]:
        """Copy the subagents and commands where Claude Code reads user agents and commands."""
        manifest_commands = self.adapter.manifest(self.source).get("commands")
        staged: list[str] = []
        destinations: dict[str, str] = {}
        for text in self.source.texts:
            if text.type == "output_style":
                self.glue.append(GluePiece(f"output_style:{text.name}", _GLUE_REASONS["output_style"]))
                continue
            if text.type not in {"agent", "command"}:
                continue
            if text.type == "agent":
                if text.rel is None:
                    continue
                # Claude Code names a subagent by its frontmatter, so the file name only has to be safe.
                file_name = f"{safe_name(PurePosixPath(text.rel).stem)}.md"
                harness_name = text.name
                folder = "agents"
                full_rel = text.rel
            else:
                harness_name = (
                    text.name
                    if text.rel is None or isinstance(manifest_commands, dict)
                    else PurePosixPath(text.rel).stem
                )
                file_name = f"{safe_name(harness_name)}.md"
                folder = "commands"
                full_rel = text.rel if text.rel is not None else f"commands/{safe_name(text.name)}.md"
            dest_rel = f"{folder}/{file_name}"
            if dest_rel in destinations:
                raise ValueError(
                    f"Refusing to stage loose parts: '{destinations[dest_rel]}' and '{text.name}' would both be "
                    f"staged as $CLAUDE_CONFIG_DIR/{dest_rel}"
                )
            destinations[dest_rel] = text.name
            bundle_rel = f"{LOOSE_BUNDLE_DIR}/{dest_rel}"
            # A body can run a plugin file (``!`${CLAUDE_PLUGIN_ROOT}/scripts/x.sh` ``): copy what it names.
            self._note_references(text.text)
            if text.rel is None:
                self.bundle.generated[bundle_rel] = text.text
            else:
                self.bundle.files.append((self._plugin_file(text.rel, text.type), bundle_rel))
            target = f"$CLAUDE_CONFIG_DIR/{dest_rel}"
            self.setup.append(
                f"skilleval_copy {shlex.quote(f'{NATIVE_ROOT}/claude-code/loose/{dest_rel}')} {shell_path(target)}"
            )
            self.checks.append(
                CensusCheck(
                    text.type,
                    text.name,
                    "file",
                    target,
                    label=f"user {folder} listing",
                    harness_name=harness_name if text.type == "command" else "",
                )
            )
            self._place(
                f"{text.type}:{harness_name}",
                text.type,
                text.rel,
                full=f"{_PLUGIN_BUNDLE}/{full_rel}",
                loose=f"{BUNDLE_DIRNAME}/{bundle_rel}",
            )
            staged.append(text.name)
        return staged

    def _plugin_file(self, rel: str, what: str) -> Path:
        kind = self.reader.kind(PurePosixPath(rel))
        if kind != "file":
            raise ValueError(
                f"Refusing to stage plugin {what} '{rel}' for the loose-parts arm: it is not a single-link regular "
                f"file inside the plugin root ({kind})"
            )
        return self.source.plugin_root / rel

    def _note_references(self, value: Any) -> None:
        for text in _strings(value):
            for match in _ROOT_PATH_RE.finditer(text):
                self.references.append(match.group(1) or "")

    def _scripts(self) -> None:
        """Copy every plugin file a staged hook or MCP server references through the plugin root."""
        rels: set[str] = set()
        for raw in self.references:
            rel = posixpath.normpath(raw.strip("/")) if raw.strip("/") else "."
            rels.add(rel)
        if "." in rels:
            self._whole_root()
            return
        files: dict[str, Path] = {}
        trees: list[str] = []
        for rel in sorted(rels):
            if rel == ".." or rel.startswith("../"):
                self.not_staged.append(
                    GluePiece(
                        f"script:{rel}",
                        "points outside the plugin root; neither arm stages it (the plugin arm's copy holds only the "
                        "plugin root)",
                    )
                )
                continue
            if any(rel.startswith(f"{tree}/") for tree in trees):
                continue
            if not self._kept_by_plugin_copy(rel):
                self.not_staged.append(
                    GluePiece(f"script:{rel}", "the plugin arm's copy leaves this path out, so the loose arm does too")
                )
                continue
            kind = self.reader.kind(PurePosixPath(rel))
            if kind == "missing":
                self.not_staged.append(
                    GluePiece(f"script:{rel}", "the referenced path does not exist in the plugin; neither arm has it")
                )
                continue
            if kind == "file":
                files[rel] = self.source.plugin_root / rel
            elif kind == "dir":
                trees.append(rel)
                files.update(self._tree_files(rel))
            else:
                raise ValueError(
                    f"Refusing to stage referenced plugin file '{rel}' for the loose-parts arm: it is a link or "
                    f"special file ({kind})"
                )
            exclude = tuple(path for path in self.generated_in_plugin if path == rel or path.startswith(f"{rel}/"))
            self._place(
                f"script:{rel}",
                "script",
                rel,
                full=f"{_PLUGIN_BUNDLE}/{rel}",
                loose=f"{_LOOSE_BUNDLE}/root/{rel}",
                exclude=tuple(path.removeprefix(rel).lstrip("/") for path in exclude),
            )
        for rel, path in sorted(files.items()):
            self.bundle.files.append((path, f"{LOOSE_BUNDLE_DIR}/root/{rel}"))

    def _whole_root(self) -> None:
        """A hook or server names the plugin root itself: copy the plugin tree there as the plugin arm does."""
        self.bundle.plugin_tree = f"{LOOSE_BUNDLE_DIR}/root"
        self._place(
            "script:.",
            "script",
            ".",
            full=_PLUGIN_BUNDLE,
            loose=f"{_LOOSE_BUNDLE}/root",
            exclude=self.generated_in_plugin,
        )

    def _kept_by_plugin_copy(self, rel: str) -> bool:
        """Whether the plugin arm's whole-plugin copy keeps ``rel`` (same ignore rules as ``stage_native_bundle``)."""
        from skillevaluator.tier3.harbor.native_staging import _plugin_tree_ignore

        ignore = _plugin_tree_ignore(self.source)
        directory = self.source.plugin_root
        for part in PurePosixPath(rel).parts:
            if part in ignore(str(directory), [part]):
                return False
            directory = directory / part
        return True

    def _tree_files(self, rel: str) -> dict[str, Path]:
        """The regular files under a referenced plugin directory that the plugin arm's copy keeps."""
        from skillevaluator.tier3.harbor.native_staging import _plugin_tree_ignore

        ignore = _plugin_tree_ignore(self.source)
        root = self.source.plugin_root
        found: dict[str, Path] = {}
        pending = [PurePosixPath(rel)]
        while pending:
            current = pending.pop()
            directory = root / current.as_posix()
            try:
                with os.scandir(directory) as iterator:
                    names = sorted(entry.name for entry in iterator)
            except OSError as exc:
                raise ValueError(f"Refusing to stage referenced plugin directory '{current}': {exc}") from exc
            skipped = set(ignore(str(directory), names))
            for name in names:
                if name in skipped:
                    continue
                child = current / name
                kind = self.reader.kind(child)
                if kind == "dir":
                    pending.append(child)
                elif kind == "file":
                    found[child.as_posix()] = root / child.as_posix()
                else:
                    raise ValueError(
                        f"Refusing to stage referenced plugin file '{child}' for the loose-parts arm: it is a link or "
                        f"special file ({kind})"
                    )
                if len(found) > _MAX_REFERENCED_FILES:
                    raise ValueError(f"Refusing to stage referenced plugin directory '{rel}': too many files")
        return found

    def _glue(self) -> None:
        for name in sorted(self.source.lsp_servers):
            self.glue.append(GluePiece(f"lsp:{name}", _GLUE_REASONS["lsp"]))
        for key in sorted(self.source.settings):
            self.glue.append(GluePiece(f"settings:{key}", _GLUE_REASONS["settings"]))
        for component_type, name in self.source.other:
            if component_type == "bin":
                self.glue.append(GluePiece("bin:bin/", _GLUE_REASONS["bin"]))
            elif (component_type == "lsp" and name in self.source.lsp_servers) or (
                component_type == "settings" and self.source.settings
            ):
                continue
            else:
                self.not_staged.append(
                    GluePiece(
                        f"{component_type}:{name}",
                        "not staged in any arm: the Claude Code plugin arm does not load it either",
                    )
                )
        self.glue.append(
            GluePiece(f"manifest:{self.source.manifest_rel or '.claude-plugin/plugin.json'}", _GLUE_REASONS["manifest"])
        )
        user_config = self.source.manifest.get("userConfig") if self.source.contained else None
        if isinstance(user_config, Mapping):
            for key in sorted(str(item) for item in user_config):
                self.glue.append(GluePiece(f"userConfig:{key}", _GLUE_REASONS["userConfig"]))


def _skill_manifest_text(skill_dir: Path) -> str:
    """A member skill's ``SKILL.md`` (bounded, no-follow read), or ``""`` when it has none."""
    reader = PluginRootReader(skill_dir)
    for name in ("SKILL.md", "skill.md"):
        if reader.kind(PurePosixPath(name)) == "file":
            return reader.read_text(PurePosixPath(name), CONTENT_DEDUP_MAX_FILE_BYTES)
    return ""


def _plugin_rel(plugin_root: Path, path: Path) -> str | None:
    try:
        return path.resolve().relative_to(plugin_root.resolve()).as_posix()
    except ValueError:
        return None


def _rule_source(source: NativePluginSource, name: str) -> str | None:
    """The plugin-relative rule file, when the rule name is one (``rules/<name>``)."""
    candidate = PurePosixPath("rules") / name
    reader = PluginRootReader(source.plugin_root)
    return candidate.as_posix() if reader.kind(candidate) == "file" else None


def _stage_member_skills(skills_dir: Path, member_skills: Sequence[Path], excluded_roots: Sequence[Path]) -> None:
    """The task skills projection of the member skills, with the emitter's own copy rules."""
    from skillevaluator.tier3.harbor.adapter import _runtime_skill_copy_ignore, _sanitize_staged_runtime_skills
    from skillevaluator.tier3.harbor.secure_copy import copytree_secure

    skills_dir.mkdir(parents=True, exist_ok=True)
    for skill in member_skills:
        if not skill.is_dir():
            continue
        destination = skills_dir / skill.name
        if destination.exists():
            continue
        copytree_secure(
            skill, destination, dirs_exist_ok=True, ignore=_runtime_skill_copy_ignore(skill, excluded_roots)
        )
    _sanitize_staged_runtime_skills(skills_dir)


# --------------------------------------------------------------------------- #
# Hashing                                                                      #
# --------------------------------------------------------------------------- #
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(_canonical_roots(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _read_regular(path: Path) -> bytes | None:
    try:
        metadata = path.lstat()
    except OSError:
        return None
    if not stat.S_ISREG(metadata.st_mode):
        return None
    return path.read_bytes()


def _tree_digest(root: Path, overrides: Mapping[str, Path], exclude: Sequence[str]) -> str | None:
    """sha256 over ``<path>\\t<file sha256>\\t<x|->`` lines of every file under ``root`` (no link followed)."""
    try:
        if not stat.S_ISDIR(root.lstat().st_mode):
            return None
    except OSError:
        return None
    lines: list[str] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for name in sorted(filenames) + [item for item in dirnames if (Path(directory) / item).is_symlink()]:
            path = Path(directory) / name
            rel = path.relative_to(root).as_posix()
            if any(rel == skipped or rel.startswith(f"{skipped}/") for skipped in exclude):
                continue
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                lines.append(f"{rel}\tnot-a-regular-file\n")
                continue
            original = overrides.get(path.as_posix())
            data = original.read_bytes() if original is not None else path.read_bytes()
            executable = "x" if metadata.st_mode & stat.S_IXUSR else "-"
            lines.append(f"{rel}\t{_sha256(data)}\t{executable}\n")
    return _sha256("".join(sorted(lines)).encode())


def _load_json(env_dir: Path, rel: str, cache: dict[str, Any]) -> Any:
    if rel not in cache:
        data = _read_regular(env_dir / rel)
        try:
            cache[rel] = json.loads(data.decode("utf-8")) if data is not None else None
        except (UnicodeDecodeError, ValueError):
            cache[rel] = None
    return cache[rel]


def _hook_owner(handler: Any) -> str | None:
    """The hook source a census-wrapped command handler belongs to (from its hook id)."""
    if not isinstance(handler, Mapping):
        return None
    identifier: str | None = None
    command = handler.get("command")
    args = handler.get("args")
    if command == "/bin/sh" and isinstance(args, list) and len(args) > 1 and args[0] == HOOK_CENSUS_SCRIPT:
        identifier = str(args[1])
    elif isinstance(command, str) and command.startswith(f"/bin/sh {HOOK_CENSUS_SCRIPT} "):
        try:
            words = shlex.split(command)
        except ValueError:
            return None
        identifier = words[2] if len(words) > 2 else None
    match = _HOOK_ID_SOURCE_RE.match(identifier or "")
    return match.group("source") if match else None


def _hook_groups(config: Any, source_name: str) -> list[Any] | None:
    """``[event, group]`` pairs of one hook source, in staged order (a group belongs to its first id's source)."""
    events = config.get("hooks") if isinstance(config, Mapping) else None
    if not isinstance(events, Mapping):
        return None
    owned: list[Any] = []
    for event, groups in events.items():
        for group in groups if isinstance(groups, list) else ():
            handlers = group.get("hooks") if isinstance(group, Mapping) else None
            owners = [owner for handler in handlers or () if (owner := _hook_owner(handler)) is not None]
            if (owners[0] if owners else "") == source_name:
                owned.append([event, group])
    return owned or None


def _piece_digest(env_dir: Path, placement: _Placement, cache: dict[str, Any]) -> str | None:
    """One piece's sha256 in one staged arm, or ``None`` when the arm does not have it."""
    if placement.kind == "mcp":
        config = _load_json(env_dir, placement.path, cache)
        servers = config.get("mcpServers") if isinstance(config, Mapping) else None
        entry = servers.get(placement.key) if isinstance(servers, Mapping) else None
        return _sha256(_canonical_json(entry)) if isinstance(entry, Mapping) else None
    if placement.kind == "hook":
        groups = _hook_groups(_load_json(env_dir, placement.path, cache), placement.key or "")
        return _sha256(_canonical_json(groups)) if groups is not None else None
    target = env_dir / placement.path
    overrides = {(env_dir / staged).as_posix(): original for staged, original in placement.overrides}
    try:
        is_dir = stat.S_ISDIR(target.lstat().st_mode)
    except OSError:
        return None
    if is_dir:
        return _tree_digest(target, overrides, placement.exclude)
    original = overrides.get(target.as_posix())
    data = original.read_bytes() if original is not None else _read_regular(target)
    return _sha256(data) if data is not None else None


def compare_content_hashes(hashes_by_arm: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    """``content_hashes``, ``equal_across_arms``, and ``mismatches`` for the staged arms' piece hashes.

    A piece is a mismatch when the full and loose arms disagree on it (or only
    one has it), or when the skills-only arm's copy of a skill differs from the
    others, misses a skill they have, or holds a non-skill piece. Glue is never
    hashed, and names are bare ids, so namespacing and packaging never count.
    """
    hashes = {_arm_key(arm): dict(sorted(values.items())) for arm, values in hashes_by_arm.items()}
    full, loose, skills = hashes.get(ARM_WITH), hashes.get(LOOSE_PARTS_ARM), hashes.get(ARM_SUM_OF_PARTS)
    mismatches: set[str] = set()
    if full is not None and loose is not None:
        mismatches.update(piece for piece in full.keys() | loose.keys() if full.get(piece) != loose.get(piece))
    if skills is not None:
        for piece, digest in skills.items():
            if not piece.startswith("skill:"):
                mismatches.add(piece)
            for other in (full, loose):
                if other is not None and other.get(piece) != digest:
                    mismatches.add(piece)
        reference = loose if loose is not None else full
        mismatches.update(piece for piece in reference or () if piece.startswith("skill:") and piece not in skills)
    return {
        "content_hashes": dict(sorted(hashes.items())),
        "equal_across_arms": not mismatches,
        "mismatches": sorted(mismatches),
    }


# --------------------------------------------------------------------------- #
# Load evidence (Claude Code system/init, MCP call evidence)                    #
# --------------------------------------------------------------------------- #
def _init_names(events: Sequence[Mapping[str, Any]], key: str) -> set[str]:
    return {item for event in events for item in event.get(key) or () if isinstance(item, str)}


def _mcp_statuses(events: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Server name -> the best status any init event reported (``connected`` wins)."""
    statuses: dict[str, str] = {}
    for event in events:
        for entry in event.get("mcp_servers") or ():
            if isinstance(entry, Mapping) and isinstance(entry.get("name"), str):
                status = str(entry.get("status") or "unknown")[:64]
                if statuses.get(entry["name"]) != "connected":
                    statuses[entry["name"]] = status
    return statuses


def _successful_mcp_calls(trajectory: Any, servers: Sequence[str], plugin: str) -> dict[str, dict[str, Any]]:
    """Per declared server, the MCP call outcome counts the plugin signals read from the trajectory."""
    if not isinstance(trajectory, Mapping) or not servers:
        return {}
    from skillevaluator.tier3.eval_core.plugin_signals import compute_plugin_signals

    signals = compute_plugin_signals(trajectory, None, declared={"mcp": list(servers), "plugin": [plugin]})
    by_server = (signals or {}).get("mcp_calls", {}).get("by_server", {})
    return {str(name): dict(counts) for name, counts in by_server.items() if isinstance(counts, Mapping)}


def loose_parts_load_evidence(
    system_init_events: Sequence[Mapping[str, Any] | None],
    trajectory: Mapping[str, Any] | None,
    loose_parts: LooseParts,
    *,
    arm: str = LOOSE_PARTS_ARM,
) -> dict[str, dict[str, Any]]:
    """``{piece id: {loaded, source, evidence, ...}}`` for one trial of ``arm``.

    Only Claude Code's own ``system/init`` report marks a piece loaded
    (``source: system/init``): a skill in ``skills``/``slash_commands``, a
    subagent in ``agents``, a command in ``slash_commands``, an MCP server in
    ``mcp_servers`` with status ``connected``. The loose arm and the
    skills-only arm use bare names; the full plugin arm the plugin's
    namespaced names (``<plugin>:<name>``, ``plugin:<plugin>:<server>``) and the
    plugin must be listed. Workaround for servers init reports ``pending``
    (slow start) or not at all: a successful call of one of the server's tools
    in the trial's ATIF trajectory marks it loaded (``source: mcp-call``).
    Rules, hooks, and referenced scripts are not in the startup report:
    ``loaded`` is ``None`` (``source: unverifiable``), never a load failure.
    """
    arm = _arm_key(arm)
    events = [event for event in system_init_events if isinstance(event, Mapping)]
    placements = loose_parts._placements(arm)
    source = loose_parts.source
    plugin = plugin_slug(source.plugin_name) if source is not None else ""
    full = arm == ARM_WITH
    names = {
        "skill": _init_names(events, "skills") | _init_names(events, "slash_commands"),
        "agent": _init_names(events, "agents"),
        "command": _init_names(events, "slash_commands"),
    }
    statuses = _mcp_statuses(events)
    tools = _init_names(events, "tools")
    plugin_listed = any(
        isinstance(entry, Mapping) and entry.get("name") == plugin
        for event in events
        for entry in event.get("plugins") or ()
    )
    servers = [placement.name for placement in placements if placement.kind == "mcp"]
    calls = _successful_mcp_calls(trajectory, servers, plugin)
    evidence: dict[str, dict[str, Any]] = {}
    for placement in placements:
        kind, name = placement.kind, placement.name
        if kind not in _INIT_KINDS:
            evidence[placement.id] = {
                "loaded": None,
                "source": _EVIDENCE_NONE,
                "evidence": f"Claude Code's system/init event does not report {kind}s; staged file listing only",
            }
            continue
        if not events:
            evidence[placement.id] = {
                "loaded": False,
                "source": _EVIDENCE_NONE,
                "evidence": "no system/init event in the Claude Code log (the agent did not start)",
            }
            continue
        if full and not plugin_listed:
            evidence[placement.id] = {
                "loaded": False,
                "source": _EVIDENCE_INIT,
                "evidence": f"claude-code did not load plugin {plugin} (not listed in its init event)",
            }
            continue
        if kind == "mcp":
            evidence[placement.id] = _mcp_evidence(name, plugin if full else "", statuses, tools, calls)
            continue
        reported = f"{plugin}:{name}" if full else name
        if reported in names[kind]:
            evidence[placement.id] = {"loaded": True, "source": _EVIDENCE_INIT, "evidence": f"init lists {reported}"}
        else:
            evidence[placement.id] = {
                "loaded": False,
                "source": _EVIDENCE_INIT,
                "evidence": f"not reported by claude-code init (no {reported})",
            }
    return evidence


def _mcp_evidence(
    server: str,
    plugin: str,
    statuses: Mapping[str, str],
    tools: set[str],
    calls: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    reported = f"plugin:{plugin}:{server}" if plugin else server
    spelled = _SERVER_NAME_RE.sub("_", f"plugin_{plugin}_{server}" if plugin else server)
    prefix = f"mcp__{spelled}__"
    listed_tools = sorted(tool[len(prefix) :] for tool in tools if tool.startswith(prefix))
    status = statuses.get(reported)
    row: dict[str, Any] = {"status": status, "tools": listed_tools}
    if status == "connected":
        return {"loaded": True, "source": _EVIDENCE_INIT, "evidence": f"init reports {reported} connected", **row}
    said = f"init reported {reported} {status}" if status else f"init did not report {reported}"
    counts = calls.get(server) or {}
    succeeded = counts.get("succeeded") if isinstance(counts.get("succeeded"), int) else 0
    if succeeded:
        called = sorted(
            label for label, outcome in (counts.get("by_tool") or {}).items() if (outcome or {}).get("succeeded")
        )
        return {
            "loaded": True,
            "source": _EVIDENCE_CALL,
            "evidence": f"{said}; {succeeded} successful call(s) of {', '.join(called) or server} in the trajectory",
            **row,
        }
    attempted = counts.get("total") if isinstance(counts.get("total"), int) else 0
    note = f"; {attempted} call(s), none succeeded" if attempted else "; no successful call of its tools"
    return {"loaded": False, "source": _EVIDENCE_INIT, "evidence": f"{said}{note}", **row}


def load_failures_by_case(
    evidence_by_arm: Mapping[str, Mapping[str, Sequence[Mapping[str, Mapping[str, Any]]]]],
) -> dict[str, list[dict[str, Any]]]:
    """``{case id: [{arm, attempt, piece, reason}]}`` for every case where a staged piece did not load.

    ``evidence_by_arm`` maps an arm to ``{case id: [evidence per attempt]}``
    (each from :func:`loose_parts_load_evidence`). A piece with ``loaded`` False
    in any attempt of any arm makes the case a load failure; the stats drop that
    case from every arm, so the arms stay paired. ``None`` (not verifiable) is
    never a failure.
    """
    failures: dict[str, list[dict[str, Any]]] = {}
    for arm, cases in sorted(evidence_by_arm.items()):
        for case_id, attempts in sorted(cases.items()):
            for attempt, evidence in enumerate(attempts, start=1):
                for piece, row in sorted(evidence.items()):
                    if isinstance(row, Mapping) and row.get("loaded") is False:
                        failures.setdefault(str(case_id), []).append(
                            {
                                "arm": _arm_key(arm),
                                "attempt": attempt,
                                "piece": piece,
                                "reason": str(row.get("evidence") or "not loaded"),
                            }
                        )
    return failures
