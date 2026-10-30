#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Harbor Skill Evaluation Verifier -- standalone.

Reads:
  /logs/agent/trajectory.json   -- ATIF trajectory from any agent (preferred)
  /logs/agent/claude-code.txt   -- Claude Code stream JSONL fallback (synthetic ATIF)
  /logs/agent/cursor-cli.txt    -- Cursor CLI stdout fallback (heuristic synthetic ATIF)
  /tests/entry.json             -- dataset entry with expected_skill, expected_behavior, etc.

Writes:
  /logs/verifier/reward.json       -- Harbor-safe numeric scores
  /logs/verifier/skill_evaluator_reward.json  -- rich SkillEvaluator scores + details
  /logs/verifier/reward.txt        -- overall score (0.0-1.0)

LLM judges use the configured public provider environment variables.
RAGAS is used for goal_accuracy and accuracy when available.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import ipaddress
import json
import logging
import math
import os
import random
import re
import signal
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from contextvars import ContextVar
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import quote, unquote_to_bytes, urlparse, urlsplit

import idna

_SCRIPT_TESTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_TESTS_DIR))
try:
    from log_converters import load_trajectory_with_fallback
except ImportError:  # pragma: no cover -- older task bundles

    def load_trajectory_with_fallback(trajectory_path, logs_dir=None):
        _ = logs_dir  # full implementation reads sibling logs; stub is trajectory.json only
        meta: dict[str, Any] = {"source": None, "warning": None, "note": None}
        if trajectory_path.exists():
            try:
                data = json.loads(trajectory_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("steps"):
                    meta["source"] = "trajectory.json"
                    return data, meta
            except (json.JSONDecodeError, OSError) as e:
                meta["warning"] = str(e)
        return None, meta


try:
    import codex_tool_call_normalizer as _tool_normalizer
except ImportError:  # pragma: no cover -- source-tree import only
    from skillevaluator.tier3.eval_core import codex_tool_call_normalizer as _tool_normalizer

atif_content_has_text = _tool_normalizer.atif_content_has_text
atif_content_text = _tool_normalizer.atif_content_text
iter_normalized_tool_calls = _tool_normalizer.iter_normalized_tool_calls
normalized_tool_call_error_flag = _tool_normalizer.normalized_tool_call_error_flag
normalized_tool_call_is_error = _tool_normalizer.normalized_tool_call_is_error
normalized_tool_call_observation = _tool_normalizer.normalized_tool_call_observation
normalized_tool_call_wrapper_observation = _tool_normalizer.normalized_tool_call_wrapper_observation

try:
    import evidence as _evidence
except ImportError:  # pragma: no cover -- source-tree import only
    from skillevaluator import evidence as _evidence

evidence_ref_identity = _evidence.evidence_ref_identity

try:
    from character_facts import append_character_facts, character_facts_enabled
except ImportError:  # pragma: no cover -- source-tree import only
    from skillevaluator.tier3.eval_core.character_facts import append_character_facts, character_facts_enabled


_CHECKS_PATH = _SCRIPT_TESTS_DIR / "checks.py"
if _CHECKS_PATH.is_file():
    # A host process may already have an unrelated module named "checks".
    # Reuse only this bundle's module; otherwise load its exact file without
    # replacing another caller's module cache. Import errors must propagate.
    _shared_checks = sys.modules.get("checks")
    if getattr(_shared_checks, "__file__", None) != str(_CHECKS_PATH):
        _checks_spec = importlib.util.spec_from_file_location("checks", _CHECKS_PATH)
        assert _checks_spec and _checks_spec.loader
        _shared_checks = importlib.util.module_from_spec(_checks_spec)
        _checks_spec.loader.exec_module(_shared_checks)
else:  # Source-tree template: the canonical module lives in the package.
    from skillevaluator.tier3.eval_core import checks as _shared_checks

AMBIGUOUS_OUTER_EXEC_OBSERVATION = _shared_checks.AMBIGUOUS_OUTER_EXEC_OBSERVATION
UNOBSERVED_INNER_CALL = _shared_checks.UNOBSERVED_INNER_CALL
UNSUPPORTED_NATIVE_CODEX_EXEC = _shared_checks.UNSUPPORTED_NATIVE_CODEX_EXEC

# Existing names with compatible argument binding and behavior.
# Include private names: downstream differential tools and tests use them.
_accepted_skill_names = _shared_checks._accepted_skill_names
_allowed_workspace_skills = _shared_checks._allowed_workspace_skills
_apply_arithmetic_assignments = _shared_checks._apply_arithmetic_assignments
_apply_binding_builtin = _shared_checks._apply_binding_builtin
_apply_eval_bindings = _shared_checks._apply_eval_bindings
_apply_patch_call = _shared_checks._apply_patch_call
_apply_patch_command_evidence = _shared_checks._apply_patch_command_evidence
_apply_patch_protected_entry = _shared_checks._apply_patch_protected_entry
_apply_set_options = _shared_checks._apply_set_options
_arithmetic_close = _shared_checks._arithmetic_close
_arithmetic_names = _shared_checks._arithmetic_names
_arithmetic_texts = _shared_checks._arithmetic_texts
_attribute_changes = _shared_checks._attribute_changes
_bind = _shared_checks._bind
_bind_unsettled = _shared_checks._bind_unsettled
_binding_readings = _shared_checks._binding_readings
_carries_a_nested_invocation = _shared_checks._carries_a_nested_invocation
_child_environment = _shared_checks._child_environment
_cmd_executes_script = _shared_checks._cmd_executes_script
_cmd_reads_skill_md = _shared_checks._cmd_reads_skill_md
_command_input_args = _shared_checks._command_input_args
_command_names_script = _shared_checks._command_names_script
_command_position_words = _shared_checks._command_position_words
_command_reads_skill_md_arg = _shared_checks._command_reads_skill_md_arg
_command_run_before_refusal = _shared_checks._command_run_before_refusal
_command_start = _shared_checks._command_start
_compound_after_assignment = _shared_checks._compound_after_assignment
_compound_closer_ends = _shared_checks._compound_closer_ends
_declaration_options = _shared_checks._declaration_options
_double_parens_are_arithmetic = _shared_checks._double_parens_are_arithmetic
_failed_single_skill_read = _shared_checks._failed_single_skill_read
_has_literal_secret = _shared_checks._has_literal_secret
_has_unquoted_secret_var = _shared_checks._has_unquoted_secret_var
_has_unsupported_native_codex_call = _shared_checks._has_unsupported_native_codex_call
_heredoc_body_spans = _shared_checks._heredoc_body_spans
_heredoc_header = _shared_checks._heredoc_header
_heredoc_operator_index = _shared_checks._heredoc_operator_index
_integer_may_fail = _shared_checks._integer_may_fail
_interpreter_operands = _shared_checks._interpreter_operands
_interpreter_script_arg = _shared_checks._interpreter_script_arg
_is_apply_patch_action = _shared_checks._is_apply_patch_action
_is_execution_action = _shared_checks._is_execution_action
_is_file_read_action = _shared_checks._is_file_read_action
_is_heredoc_redirect = _shared_checks._is_heredoc_redirect
_is_httpie_body_item = _shared_checks._is_httpie_body_item
_is_network_exfiltration_command = _shared_checks._is_network_exfiltration_command
_is_output_redirect = _shared_checks._is_output_redirect
_is_redirection_operator = _shared_checks._is_redirection_operator
_keep_attached_descriptors = _shared_checks._keep_attached_descriptors
_lexical_path_components = _shared_checks._lexical_path_components
_looks_like_refusal = _shared_checks._looks_like_refusal
_loop_header_is_empty = _shared_checks._loop_header_is_empty
_loop_header_is_unresolved = _shared_checks._loop_header_is_unresolved
_mark_literal_dollars = _shared_checks._mark_literal_dollars
_mark_quoted_newlines = _shared_checks._mark_quoted_newlines
_mark_quoted_syntax = _shared_checks._mark_quoted_syntax
_mask_non_assignment_equals = _shared_checks._mask_non_assignment_equals
_mentions_skill_artifact = _shared_checks._mentions_skill_artifact
_names_script_anywhere = _shared_checks._names_script_anywhere
_network_shell_tokens = _shared_checks._network_shell_tokens
_normalize_skill_names = _shared_checks._normalize_skill_names
_normalized_write_path = _shared_checks._normalized_write_path
_number_may_fail = _shared_checks._number_may_fail
_parse_unit_starts = _shared_checks._parse_unit_starts
_pass_to_reference = _shared_checks._pass_to_reference
_path_with_shell_cwd = _shared_checks._path_with_shell_cwd
_pipeline_last_stage_keeps_bindings = _shared_checks._pipeline_last_stage_keeps_bindings
_pipeline_upstream_names_script = _shared_checks._pipeline_upstream_names_script
_plain_pipeline = _shared_checks._plain_pipeline
_possibly_references_target_directory = _shared_checks._possibly_references_target_directory
_prefixed_scope = _shared_checks._prefixed_scope
_protected_write_entry = _shared_checks._protected_write_entry
_reader_scope = _shared_checks._reader_scope
_reads_program_from_stdin = _shared_checks._reads_program_from_stdin
_redact_network_evidence = _shared_checks._redact_network_evidence
_redirects_script_to_stdin = _shared_checks._redirects_script_to_stdin
_references_exact_target_artifact = _shared_checks._references_exact_target_artifact
_refuse_assignment = _shared_checks._refuse_assignment
_resolved_shell_arg = _shared_checks._resolved_shell_arg
_rm_group_end = _shared_checks._rm_group_end
_rm_inline_code = _shared_checks._rm_inline_code
_rm_sequential_commands = _shared_checks._rm_sequential_commands
_rm_shell_tokens = _shared_checks._rm_shell_tokens
_rm_stdin_code = _shared_checks._rm_stdin_code
_rm_stdin_codes = _shared_checks._rm_stdin_codes
_rm_symbolic_link_command = _shared_checks._rm_symbolic_link_command
_rm_symlink_call_seen = _shared_checks._rm_symlink_call_seen
_rm_symlink_creation_seen = _shared_checks._rm_symlink_creation_seen
_rm_symlink_fallback = _shared_checks._rm_symlink_fallback
_rm_targets_fallback = _shared_checks._rm_targets_fallback
_rm_targets_outside_tmp = _shared_checks._rm_targets_outside_tmp
_rm_tmp_operand_is_exempt = _shared_checks._rm_tmp_operand_is_exempt
_rm_wrapped_command_index = _shared_checks._rm_wrapped_command_index
_rm_xargs_context = _shared_checks._rm_xargs_context
_rm_xargs_fallback = _shared_checks._rm_xargs_fallback
_runs_inline_code = _shared_checks._runs_inline_code
_scope_events = _shared_checks._scope_events
_script_path_matches = _shared_checks._script_path_matches
_search_path_is_root = _shared_checks._search_path_is_root
_secret_exposure_finding = _shared_checks._secret_exposure_finding
_security_finding = _shared_checks._security_finding
_shell_c_payload = _shared_checks._shell_c_payload
_shell_c_positional = _shared_checks._shell_c_positional
_shell_executable = _shared_checks._shell_executable
_shell_substitution_payloads = _shared_checks._shell_substitution_payloads
_shell_tokens = _shared_checks._shell_tokens
_skill_match_details = _shared_checks._skill_match_details
_skill_md_arg = _shared_checks._skill_md_arg
_skill_name_from_ref = _shared_checks._skill_name_from_ref
_skip_arithmetic = _shared_checks._skip_arithmetic
_skip_transparent_prefixes = _shared_checks._skip_transparent_prefixes
_skip_wrapper_options = _shared_checks._skip_wrapper_options
_split_heredoc_body = _shared_checks._split_heredoc_body
_split_heredocs = _shared_checks._split_heredocs
_split_punctuation_runs = _shared_checks._split_punctuation_runs
_string_argument = _shared_checks._string_argument
_tool_call_shell = _shared_checks._tool_call_shell
_tool_searches_root = _shared_checks._tool_searches_root
_unit_is_complete = _shared_checks._unit_is_complete
_unquoted_separator_index = _shared_checks._unquoted_separator_index
_unresolved_value = _shared_checks._unresolved_value
_unsettle_every_binding = _shared_checks._unsettle_every_binding
_unsupported_native_codex_result = _shared_checks._unsupported_native_codex_result
_unwrap_shell_command = _shared_checks._unwrap_shell_command
_value_now = _shared_checks._value_now
_walk_for_invocation = _shared_checks._walk_for_invocation
_without_redirections = _shared_checks._without_redirections
check_error_recovery = _shared_checks.check_error_recovery
check_script_execution = _shared_checks.check_script_execution
resolve_should_trigger = _shared_checks.resolve_should_trigger
score_skill_execution = _shared_checks.score_skill_execution

# All 148 constant initializers are AST-identical at the fixed base.
ACCEPTABLE_ALTERNATE_SCORE = _shared_checks.ACCEPTABLE_ALTERNATE_SCORE
WASTE_INDICATORS = _shared_checks.WASTE_INDICATORS
_ALLEXPORT_MARK = _shared_checks._ALLEXPORT_MARK
_APPLY_PATCH_BODY_RE = _shared_checks._APPLY_PATCH_BODY_RE
_APPLY_PATCH_CD_RE = _shared_checks._APPLY_PATCH_CD_RE
_APPLY_PATCH_COMMAND_RE = _shared_checks._APPLY_PATCH_COMMAND_RE
_APPLY_PATCH_DEFAULT_WORKDIR = _shared_checks._APPLY_PATCH_DEFAULT_WORKDIR
_APPLY_PATCH_HEADER_RE = _shared_checks._APPLY_PATCH_HEADER_RE
_ARITHMETIC_ASSIGNMENT_RE = _shared_checks._ARITHMETIC_ASSIGNMENT_RE
_ARITHMETIC_EXPANSION_RE = _shared_checks._ARITHMETIC_EXPANSION_RE
_ASSIGNMENT_MASK = _shared_checks._ASSIGNMENT_MASK
_ASSIGNMENT_REFUSED = _shared_checks._ASSIGNMENT_REFUSED
_ATTACHED_DESCRIPTOR_RE = _shared_checks._ATTACHED_DESCRIPTOR_RE
_ATTACHED_REDIRECT_RE = _shared_checks._ATTACHED_REDIRECT_RE
_ATTRIBUTE_LETTERS = _shared_checks._ATTRIBUTE_LETTERS
_ATTRIBUTE_MARK = _shared_checks._ATTRIBUTE_MARK
_BAD_REFERENCE_STOPS = _shared_checks._BAD_REFERENCE_STOPS
_BASED_VALUE_RE = _shared_checks._BASED_VALUE_RE
_BUILTIN_REACHES_BUILTINS = _shared_checks._BUILTIN_REACHES_BUILTINS
_COMMAND_INTRODUCING_WORDS = _shared_checks._COMMAND_INTRODUCING_WORDS
_COMMAND_POSITION_LEADERS = _shared_checks._COMMAND_POSITION_LEADERS
_COMMAND_REACHES_BUILTINS = _shared_checks._COMMAND_REACHES_BUILTINS
_COMPOUND_CLOSERS = _shared_checks._COMPOUND_CLOSERS
_COMPOUND_OPENERS = _shared_checks._COMPOUND_OPENERS
_CURL_DATA_FLAGS = _shared_checks._CURL_DATA_FLAGS
_CURL_SHORT_OPTS_WITH_ARG = _shared_checks._CURL_SHORT_OPTS_WITH_ARG
_CURL_UPLOAD_FLAGS = _shared_checks._CURL_UPLOAD_FLAGS
_DECIMAL_VALUE_RE = _shared_checks._DECIMAL_VALUE_RE
_DECLARATION_ASSIGNS_IN_ORDER = _shared_checks._DECLARATION_ASSIGNS_IN_ORDER
_DECLARATION_BUILTINS = _shared_checks._DECLARATION_BUILTINS
_DECLARATION_KEEPS_VALUE = _shared_checks._DECLARATION_KEEPS_VALUE
_DECLARATION_OPTIONS = _shared_checks._DECLARATION_OPTIONS
_DECLARATION_OPTION_SHELLS = _shared_checks._DECLARATION_OPTION_SHELLS
_DECLARING_BUILTINS = _shared_checks._DECLARING_BUILTINS
_DESCRIPTOR_OPERATOR_RE = _shared_checks._DESCRIPTOR_OPERATOR_RE
_DESTRUCTIVE_PATTERNS = _shared_checks._DESTRUCTIVE_PATTERNS
_DURATION_ARG_RE = _shared_checks._DURATION_ARG_RE
_DYNAMIC_ARGUMENT_RE = _shared_checks._DYNAMIC_ARGUMENT_RE
_EMPTY_REFERENCE_SHELLS = _shared_checks._EMPTY_REFERENCE_SHELLS
_EXECUTION_TOOL_HINTS = _shared_checks._EXECUTION_TOOL_HINTS
_EXPORTED_MARK = _shared_checks._EXPORTED_MARK
_FILE_READ_VERBS = _shared_checks._FILE_READ_VERBS
_GLUED_AKIA_BODY = _shared_checks._GLUED_AKIA_BODY
_GLUED_KEY_BODY = _shared_checks._GLUED_KEY_BODY
_GROUPING_TOKENS = _shared_checks._GROUPING_TOKENS
_HEREDOC_DELIMITER_RE = _shared_checks._HEREDOC_DELIMITER_RE
_HEREDOC_REDIRECTS = _shared_checks._HEREDOC_REDIRECTS
_HOME_DIR_PREFIX_RE = _shared_checks._HOME_DIR_PREFIX_RE
_HOME_SHORTHAND_RE = _shared_checks._HOME_SHORTHAND_RE
_HTTPIE_BODY_FLAGS = _shared_checks._HTTPIE_BODY_FLAGS
_INERT_PRINT_COMMANDS = _shared_checks._INERT_PRINT_COMMANDS
_INERT_SHELL_PRODUCERS = _shared_checks._INERT_SHELL_PRODUCERS
_INLINE_CODE = _shared_checks._INLINE_CODE
_INPUT_REDIRECTS = _shared_checks._INPUT_REDIRECTS
_INPUT_REDIRECT_OPERATORS = _shared_checks._INPUT_REDIRECT_OPERATORS
_INTEGER_ERROR_GOES_ON = _shared_checks._INTEGER_ERROR_GOES_ON
_INTEGER_VALUE_RE = _shared_checks._INTEGER_VALUE_RE
_INTERPRETER_GRAMMARS = _shared_checks._INTERPRETER_GRAMMARS
_JUSTIFYING_SHELLS = _shared_checks._JUSTIFYING_SHELLS
_LAST_STAGE_IN_CURRENT_SHELL = _shared_checks._LAST_STAGE_IN_CURRENT_SHELL
_LAST_STAGE_IN_SUBSHELL = _shared_checks._LAST_STAGE_IN_SUBSHELL
_LISTING_ACTS_ON_NAMES = _shared_checks._LISTING_ACTS_ON_NAMES
_LITERAL_DOLLAR = _shared_checks._LITERAL_DOLLAR
_LOOP_HEADER_WORDS = _shared_checks._LOOP_HEADER_WORDS
_MAX_HEREDOCS_PER_LINE = _shared_checks._MAX_HEREDOCS_PER_LINE
_MAX_HEREDOC_OPERANDS = _shared_checks._MAX_HEREDOC_OPERANDS
_MAX_NETWORK_ACTION_CHARS = _shared_checks._MAX_NETWORK_ACTION_CHARS
_MAX_PARSE_UNIT_LINES = _shared_checks._MAX_PARSE_UNIT_LINES
_MAX_PARSE_UNIT_READ = _shared_checks._MAX_PARSE_UNIT_READ
_MAX_SHELL_REFERENCE_CHARS = _shared_checks._MAX_SHELL_REFERENCE_CHARS
_MAX_SHELL_REFERENCE_DEPTH = _shared_checks._MAX_SHELL_REFERENCE_DEPTH
_MAX_SHELL_REFERENCE_TOKENS = _shared_checks._MAX_SHELL_REFERENCE_TOKENS
_MAX_SHELL_WRAPPERS = _shared_checks._MAX_SHELL_WRAPPERS
_NAMEREF_SHELLS = _shared_checks._NAMEREF_SHELLS
_NEGATIVE_NUMBER_RE = _shared_checks._NEGATIVE_NUMBER_RE
_NESTED_QUOTING_RE = _shared_checks._NESTED_QUOTING_RE
_NETWORK_CLIENT_FAST_PATTERN = _shared_checks._NETWORK_CLIENT_FAST_PATTERN
_NETWORK_CLIENT_PATTERN = _shared_checks._NETWORK_CLIENT_PATTERN
_NETWORK_EXECUTABLES = _shared_checks._NETWORK_EXECUTABLES
_NOT_A_SIMPLE_COMMAND = _shared_checks._NOT_A_SIMPLE_COMMAND
_NO_HERE_STRING_SHELLS = _shared_checks._NO_HERE_STRING_SHELLS
_NO_SCRIPT = _shared_checks._NO_SCRIPT
_NUMERIC_ATTRIBUTES = _shared_checks._NUMERIC_ATTRIBUTES
_NUMERIC_ATTRIBUTE_STOPS = _shared_checks._NUMERIC_ATTRIBUTE_STOPS
_OPAQUE_SHELL_BUILTINS = _shared_checks._OPAQUE_SHELL_BUILTINS
_OPENERS_REFUSED_AFTER_ASSIGNMENT = _shared_checks._OPENERS_REFUSED_AFTER_ASSIGNMENT
_OPERATOR_TOKEN_RE = _shared_checks._OPERATOR_TOKEN_RE
_OUTPUT_REDIRECTS = _shared_checks._OUTPUT_REDIRECTS
_PARSES_WHOLE_TEXT_FIRST = _shared_checks._PARSES_WHOLE_TEXT_FIRST
_PATH_ARG_KEYS = _shared_checks._PATH_ARG_KEYS
_PIPELINE_OPTION_RE = _shared_checks._PIPELINE_OPTION_RE
_POSITIONAL_SET_RE = _shared_checks._POSITIONAL_SET_RE
_PREFIX_EXPANDS_FIRST = _shared_checks._PREFIX_EXPANDS_FIRST
_PREFIX_OUTLIVES_SPECIAL_BUILTIN = _shared_checks._PREFIX_OUTLIVES_SPECIAL_BUILTIN
_PROMPT_INJECTION_PATTERNS = _shared_checks._PROMPT_INJECTION_PATTERNS
_PUNCTUATION_PIECE_RE = _shared_checks._PUNCTUATION_PIECE_RE
_PUNCTUATION_RUN_RE = _shared_checks._PUNCTUATION_RUN_RE
_QUOTED_NEWLINE = _shared_checks._QUOTED_NEWLINE
_QUOTED_SYNTAX_MARK = _shared_checks._QUOTED_SYNTAX_MARK
_READONLY_ATTRIBUTES = _shared_checks._READONLY_ATTRIBUTES
_READONLY_MARK = _shared_checks._READONLY_MARK
_READS_ARGV0_RE = _shared_checks._READS_ARGV0_RE
_READS_POSITIONAL_RE = _shared_checks._READS_POSITIONAL_RE
_READ_TOOL_HINTS = _shared_checks._READ_TOOL_HINTS
_REDIRECTION_OPERATOR_RE = _shared_checks._REDIRECTION_OPERATOR_RE
_REDIRECTION_WORD_RE = _shared_checks._REDIRECTION_WORD_RE
_REFERENCE_MARK = _shared_checks._REFERENCE_MARK
_REFERENCE_SHELLS = _shared_checks._REFERENCE_SHELLS
_REFUSED_ASSIGNMENT_GOES_ON = _shared_checks._REFUSED_ASSIGNMENT_GOES_ON
_RUNNER_COMMAND_PREFIXES = _shared_checks._RUNNER_COMMAND_PREFIXES
_RUNTIME_BINDING_BUILTINS = _shared_checks._RUNTIME_BINDING_BUILTINS
_SCRIPT = _shared_checks._SCRIPT
_SCRIPT_INTERPRETERS = _shared_checks._SCRIPT_INTERPRETERS
_SECRET_PATTERNS = _shared_checks._SECRET_PATTERNS
_SECRET_VAR_NAME_RE = _shared_checks._SECRET_VAR_NAME_RE
_SENSITIVE_WRITE_PATHS = _shared_checks._SENSITIVE_WRITE_PATHS
_SHELL_ASSIGNMENT_RE = _shared_checks._SHELL_ASSIGNMENT_RE
_SHELL_COMMAND_INTERPRETERS = _shared_checks._SHELL_COMMAND_INTERPRETERS
_SHELL_EXPANSION_CHARS = _shared_checks._SHELL_EXPANSION_CHARS
_SHELL_METACHARS = _shared_checks._SHELL_METACHARS
_SHELL_NAME_RE = _shared_checks._SHELL_NAME_RE
_SHELL_SEPARATORS = _shared_checks._SHELL_SEPARATORS
_SHELL_VARIABLE_RE = _shared_checks._SHELL_VARIABLE_RE
_SOURCING_COMMANDS = _shared_checks._SOURCING_COMMANDS
_SPECIAL_BUILTINS = _shared_checks._SPECIAL_BUILTINS
_SPECIAL_BUILTIN_ERROR_STOPS = _shared_checks._SPECIAL_BUILTIN_ERROR_STOPS
_STDIN_ARGV_RE = _shared_checks._STDIN_ARGV_RE
_STDIN_DEPENDENT_WRAPPERS = _shared_checks._STDIN_DEPENDENT_WRAPPERS
_SYNTAX_SHAPE_RE = _shared_checks._SYNTAX_SHAPE_RE
_SYNTAX_WORDS = _shared_checks._SYNTAX_WORDS
_TRANSPARENT_COMMAND_PREFIXES = _shared_checks._TRANSPARENT_COMMAND_PREFIXES
_UNAUTHORIZED_PATHS = _shared_checks._UNAUTHORIZED_PATHS
_UNDECIDABLE = _shared_checks._UNDECIDABLE
_UNIT_CLOSERS = _shared_checks._UNIT_CLOSERS
_UNIT_NEWLINE = _shared_checks._UNIT_NEWLINE
_UNMODELLED_CONTROL_WORDS = _shared_checks._UNMODELLED_CONTROL_WORDS
_UNRESOLVED_ARG_RE = _shared_checks._UNRESOLVED_ARG_RE
_UNSAFE_HTTP_METHODS = _shared_checks._UNSAFE_HTTP_METHODS
_UNSETTLED_EXPANSION_RE = _shared_checks._UNSETTLED_EXPANSION_RE
_UNSETTLED_VALUE = _shared_checks._UNSETTLED_VALUE
_UNSET_REFERENCE_SHELLS = _shared_checks._UNSET_REFERENCE_SHELLS
_VERSION_SUFFIX_RE = _shared_checks._VERSION_SUFFIX_RE
_WGET_DATA_FLAGS = _shared_checks._WGET_DATA_FLAGS
_WRAPPER_GRAMMARS = _shared_checks._WRAPPER_GRAMMARS
_WRAPPER_NONE = _shared_checks._WRAPPER_NONE
_WRAPPER_OK = _shared_checks._WRAPPER_OK
_WRAPPER_UNKNOWN = _shared_checks._WRAPPER_UNKNOWN
_WRITE_TOOL_HINTS = _shared_checks._WRITE_TOOL_HINTS

# Fifteen compatibility adapters preserve existing argument names, positional
# options, trajectory input, and output shape.


def _action_args(tc):
    return _shared_checks._action_args(tc)


def _action_text(tc):
    return _shared_checks._action_text(tc)


def _command_text(tc):
    return _shared_checks._command_text(tc)


def _extract_path(tc):
    return _shared_checks._extract_path(tc)


def _tool_mentions_any_skill(tc, expected_skill, acceptable_skills=None):
    return _shared_checks._tool_mentions_any_skill(tc, expected_skill, acceptable_skills)


def _tool_mentions_skill(tc, expected_skill):
    return _shared_checks._tool_mentions_skill(tc, expected_skill)


def _classify_skill_match(observed, expected_skill, acceptable_skills=None, fuzzy=False):
    return _shared_checks._classify_skill_match(observed, expected_skill, acceptable_skills, fuzzy=fuzzy)


def _cmd_references_exact_target(cmd, target_skill, _depth=0):
    return _shared_checks._cmd_references_exact_target(cmd, target_skill, _depth=_depth)


def _match_skill_name(observed, expected, fuzzy=False):
    # Retain the verifier's falsey fuzzy return value (including None, 0, '').
    # The host returns False instead. This tiny compatibility body is deliberate.
    if not observed or not expected:
        return False
    observed_l = str(observed).lower()
    expected_l = str(expected).lower()
    if observed_l == expected_l:
        return True
    return fuzzy and expected_l in observed_l


def check_activation(tool_calls, expected_skill, skill_tool_names=None, acceptable_skills=None):
    return _shared_checks.check_activation(
        tool_calls,
        expected_skill,
        skill_tool_names=skill_tool_names,
        acceptable_skills=acceptable_skills,
    )


def check_negative_case(tool_calls, skill_under_test, skill_tool_names=None):
    return _shared_checks.check_negative_case(tool_calls, skill_under_test, skill_tool_names=skill_tool_names)


def check_routing(
    tool_calls,
    expected_skill,
    skill_tool_names=None,
    workspace_skill_names=None,
    workspace_mode="isolated",
    acceptable_skills=None,
):
    return _shared_checks.check_routing(
        tool_calls,
        expected_skill,
        skill_tool_names=skill_tool_names,
        workspace_skill_names=workspace_skill_names,
        workspace_mode=workspace_mode,
        acceptable_skills=acceptable_skills,
    )


def check_workflow_order(tool_calls, skill_tool_names=None, expected_skill=None):
    return _shared_checks.check_workflow_order(
        tool_calls, skill_tool_names=skill_tool_names, expected_skill=expected_skill
    )


def check_security(traj, tool_calls, expected_skill=None, acceptable_skills=None):
    return _shared_checks.check_security(
        tool_calls,
        get_agent_text(traj),
        user_text="\n".join(
            atif_content_text(step.get("message")) for step in traj.get("steps", []) if step.get("source") == "user"
        ),
        expected_skill=expected_skill or "",
        acceptable_skills=acceptable_skills,
    )


def check_tool_efficiency(tool_calls, expected_skill=None, expected_script=None):
    result = _shared_checks.check_tool_efficiency(tool_calls, expected_skill, expected_script)
    result.pop("details", None)
    return result


logger = logging.getLogger(__name__)


def _env_path(name, default):
    return Path(os.environ.get(name, str(default)))


LOGS_DIR = _env_path("HARBOR_LOGS_DIR", "/logs")
AGENT_LOGS_DIR = _env_path("HARBOR_AGENT_LOGS_DIR", LOGS_DIR / "agent")
VERIFIER_DIR = _env_path("HARBOR_VERIFIER_DIR", LOGS_DIR / "verifier")
TESTS_DIR = _env_path("HARBOR_TESTS_DIR", "/tests")

ATIF_PATH = _env_path("HARBOR_ATIF_PATH", AGENT_LOGS_DIR / "trajectory.json")
ENTRY_PATH = _env_path("HARBOR_ENTRY_JSON", TESTS_DIR / "entry.json")
REWARD_JSON = _env_path("HARBOR_REWARD_JSON", VERIFIER_DIR / "reward.json")
REWARD_TXT = _env_path("HARBOR_REWARD_TXT", VERIFIER_DIR / "reward.txt")
SKILL_EVALUATOR_REWARD_JSON = _env_path(
    "HARBOR_SKILL_EVALUATOR_REWARD_JSON", VERIFIER_DIR / "skill_evaluator_reward.json"
)

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
NVIDIA_BUILD_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
# Last-resort judge per provider when no model is configured. Keep in sync with
# skillevaluator.provider_config.CHAT_DEFAULT_MODELS (sandbox template cannot
# import the package — see drift tests).
DEFAULT_JUDGE_MODELS = {
    "openai": "gpt-6.1-sol",
    "anthropic": "claude-opus-5-5",
    "nv_build": "nvidia/nemotron-3-super-120b-a12b",
    "bedrock": "us.anthropic.claude-opus-5-5",
    "openai-compatible": "nvidia/nvidia/nemotron-3-super-120b-long-ctx",
}
JUDGE_REASONING_EFFORT = "medium"
REASONING_MAX_COMPLETION_TOKENS = 32768
ANTHROPIC_REFUSAL_FALLBACK_MODEL = "claude-opus-4-8"
ANTHROPIC_SERVER_SIDE_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_ANTHROPIC_DNS_LABEL_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
_ANTHROPIC_INTERNAL_LABEL_RE = re.compile(r"^[A-Za-z0-9_](?:[A-Za-z0-9_-]{0,61}[A-Za-z0-9_])?$")
_ANTHROPIC_IPV6_ZONE_RE = re.compile(r"^[A-Za-z0-9._~-]+$")
_ANTHROPIC_PATH_SAFE = "/:@!$&'()*+,;=-._~%"
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_HEX_DIGIT_BYTES = frozenset(b"0123456789abcdefABCDEF")
_UNRESERVED_BYTES = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")

_ERROR_REDACTION_MARKER = "[REDACTED]"
_JUDGE_ERROR_REASON_LIMIT = 512
_JUDGE_TEXT_LIMIT = 512
# Shorter placeholders are not credible provider credentials and can corrupt report schema keys.
_MIN_EXACT_SECRET_LENGTH = 8
_CREDENTIAL_ENV_VARS = (
    "OPENAI_API_KEY",
    "NVIDIA_API_KEY",
    "ANTHROPIC_API_KEY",
    "SKILL_EVAL_LLM_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SECURITY_TOKEN",
    "AWS_SESSION_TOKEN",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN",
)


DEFAULT_METRIC_SET = "skill-evaluator-default-v2"
DEFAULT_SCORE_POLICY = "skill-evaluator-dimension-mean-v1"
DISPLAY_METRICS = [
    "security",
    "skill_execution",
    "skill_efficiency",
    "accuracy",
    "goal_accuracy",
    "behavior_check",
]


LOG_SK_RE = re.compile(r"(?<![A-Za-z0-9_-])sk-[a-zA-Z0-9_-]{8,}|sk-" + _GLUED_KEY_BODY)
LOG_NVAPI_RE = re.compile(r"(?<![A-Za-z0-9_-])nvapi-[a-zA-Z0-9_-]{8,}|nvapi-" + _GLUED_KEY_BODY)
LOG_CRSR_RE = re.compile(r"(?<![A-Za-z0-9_-])crsr_[a-f0-9]{16,}")
OPENSHIFT_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_-])sha256~[A-Za-z0-9._~-]+")
LOG_YA29_RE = re.compile(r"(?<![A-Za-z0-9_-])ya29\.[A-Za-z0-9_-]{20,}")
# A JWT used to start at any ``\beyJ``, so in a run of JWT characters such as
# "eyJ-" * n every "-eyJ" was a start, and each start scanned to the end of the
# run looking for ".". Now a match starts only at the beginning of a run. The part
# of the run before its first ``\beyJ`` is captured as ``lead`` and written back
# unchanged, which keeps JWTs glued to a "-" (x-eyJ...) redacted. Later starts in
# the same run are never tried: their first segment reaches the same "." with
# fewer characters, so they could only fail where the first start failed. ``lead``
# stops at the first ``\beyJ`` and every earlier offset fails ``\beyJ``, so
# backtracking into it is cheap. No atomic groups or possessive quantifiers: the
# Harbor verifier copy runs on the task image's python3, which may predate 3.11.
LOG_JWT_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?P<lead>(?:(?!\beyJ)[A-Za-z0-9_-])*)"
    r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\b"
)
# GitHub classic (ghp_/gho_/ghu_/ghs_/ghr_) and fine-grained (github_pat_) tokens.
# Single bounded character classes keep both patterns linear.
LOG_GITHUB_TOKEN_RE = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b")
LOG_GITHUB_PAT_RE = re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,255}\b")


def redact_secrets_in_log_line(line, *, extra_secret_values=None):
    """Best-effort mask common key shapes in Harbor verifier output text."""
    for secret in sorted(set(extra_secret_values or ()), key=len, reverse=True):
        if secret and len(secret) >= _MIN_EXACT_SECRET_LENGTH:
            line = line.replace(secret, "<redacted>")
    line = LOG_SK_RE.sub("sk-<redacted>", line)
    line = LOG_NVAPI_RE.sub("nvapi-<redacted>", line)
    line = LOG_CRSR_RE.sub("crsr_<redacted>", line)
    line = LOG_GITHUB_TOKEN_RE.sub(lambda match: match.group()[:4] + "<redacted>", line)
    line = LOG_GITHUB_PAT_RE.sub("github_pat_<redacted>", line)
    line = OPENSHIFT_TOKEN_RE.sub("sha256~<redacted>", line)
    line = LOG_YA29_RE.sub("ya29.<redacted>", line)
    if "eyJ" not in line:  # every JWT match contains "eyJ"; skip the scan on ordinary lines
        return line
    return LOG_JWT_RE.sub(r"\g<lead>jwt-<redacted>", line)


# ── ATIF Helpers ─────────────────────────────────────────────────────────────


# Host and container evidence use the same implementation. The staged file
# takes precedence only when it exists beside this evaluator-owned verifier.
_ATIF_HELPERS_PATH = _SCRIPT_TESTS_DIR / "atif_helpers.py"
if _ATIF_HELPERS_PATH.is_file():
    import importlib.util

    _atif_spec = importlib.util.spec_from_file_location("_skillevaluator_shared_atif", _ATIF_HELPERS_PATH)
    assert _atif_spec and _atif_spec.loader
    _shared_atif = importlib.util.module_from_spec(_atif_spec)
    _atif_spec.loader.exec_module(_shared_atif)
else:
    from skillevaluator.tier3.eval_core import atif_helpers as _shared_atif

# Retain template helper names used by graders and differential tools.
iter_tool_calls = _shared_atif.iter_tool_calls
_tool_call_observation = _shared_atif._tool_call_observation
_tool_call_wrapper_observation = _shared_atif._tool_call_wrapper_observation
get_all_tool_calls = _shared_atif.get_all_tool_calls
get_skill_tool_calls = _shared_atif.get_skill_tool_calls
get_read_calls = _shared_atif.get_read_calls
get_bash_commands = _shared_atif.get_bash_commands
get_agent_text = _shared_atif.get_agent_text
extract_tool_calls_as_dicts = _shared_atif.extract_tool_calls_as_dicts
build_conversation_summary = _shared_atif.build_conversation_summary
_BEHAVIOR_EVIDENCE_MAX_CHARS = _shared_atif._BEHAVIOR_EVIDENCE_MAX_CHARS
_DEFAULT_BEHAVIOR_FINAL_RESPONSE_LIMIT = _shared_atif._DEFAULT_BEHAVIOR_FINAL_RESPONSE_LIMIT
_DEFAULT_BEHAVIOR_CHECK_BUDGET = _shared_atif._DEFAULT_BEHAVIOR_CHECK_BUDGET
_DEFAULT_TOOL_HISTORY_HEADROOM = _shared_atif._DEFAULT_TOOL_HISTORY_HEADROOM
_MIN_BEHAVIOR_HISTORY_HEADROOM = _shared_atif._MIN_BEHAVIOR_HISTORY_HEADROOM
_SECTION_COMPACT_TOOL_HISTORY = _shared_atif._SECTION_COMPACT_TOOL_HISTORY
_SECTION_FILE_CHANGES = _shared_atif._SECTION_FILE_CHANGES
_SECTION_FINAL_RESPONSE = _shared_atif._SECTION_FINAL_RESPONSE
_SECTION_USER_REQUEST = _shared_atif._SECTION_USER_REQUEST
_env_positive_int = _shared_atif._env_positive_int
_behavior_final_response_limit = _shared_atif._behavior_final_response_limit
_behavior_check_budget = _shared_atif._behavior_check_budget
_BEHAVIOR_WRITE_TOOLS = _shared_atif._BEHAVIOR_WRITE_TOOLS
_BEHAVIOR_EXEC_TOOLS = _shared_atif._BEHAVIOR_EXEC_TOOLS
_BEHAVIOR_WRITE_COMMAND_MARKERS = _shared_atif._BEHAVIOR_WRITE_COMMAND_MARKERS
_BEHAVIOR_WRITE_REDIRECT_RE = _shared_atif._BEHAVIOR_WRITE_REDIRECT_RE
_BEHAVIOR_PYTHON_WRITE_RE = _shared_atif._BEHAVIOR_PYTHON_WRITE_RE
_TOOL_NAME_SEPARATORS = _shared_atif._TOOL_NAME_SEPARATORS
_get_final_response = _shared_atif.get_final_response
_truncate_for_behavior = _shared_atif._truncate_for_behavior
_append_section_with_budget = _shared_atif._append_section_with_budget
_tool_file_path = _shared_atif._tool_file_path
_tool_write_body = _shared_atif._tool_write_body
_command_looks_like_write = _shared_atif._command_looks_like_write
_tool_name_looks_like_write = _shared_atif._tool_name_looks_like_write
_collect_file_change_evidence = _shared_atif._collect_file_change_evidence
build_behavior_evidence = _shared_atif.build_behavior_evidence
_METRIC_EVIDENCE_REF_METRICS = _shared_atif._METRIC_EVIDENCE_REF_METRICS
_METRIC_EVIDENCE_EXCERPT_CHARS = _shared_atif._METRIC_EVIDENCE_EXCERPT_CHARS
_METRIC_EVIDENCE_MAX_TOOL_REFS = _shared_atif._METRIC_EVIDENCE_MAX_TOOL_REFS
_METRIC_EVIDENCE_MAX_FILE_REFS = _shared_atif._METRIC_EVIDENCE_MAX_FILE_REFS
_EXPECTED_ARTIFACT_PATH_RE = _shared_atif._EXPECTED_ARTIFACT_PATH_RE
_redact_evidence_text = _shared_atif._redact_evidence_text
_evidence_excerpt = _shared_atif._evidence_excerpt
_evidence_ref = _shared_atif._evidence_ref
_dedupe_evidence_refs = _shared_atif._dedupe_evidence_refs
_final_response_ref = _shared_atif._final_response_ref
_tool_call_ref = _shared_atif._tool_call_ref
_tool_call_refs = _shared_atif._tool_call_refs
_tool_observation_refs = _shared_atif._tool_observation_refs
_file_change_refs = _shared_atif._file_change_refs
_first_expected_artifact_path = _shared_atif._first_expected_artifact_path
_expected_artifact_refs = _shared_atif._expected_artifact_refs
_expected_behavior_refs = _shared_atif._expected_behavior_refs
_ground_truth_ref = _shared_atif._ground_truth_ref
build_metric_evidence_refs = _shared_atif.build_metric_evidence_refs
attach_metric_evidence_refs = _shared_atif.attach_metric_evidence_refs
_BUNDLE_ITEM_CHARS = _shared_atif._BUNDLE_ITEM_CHARS
_DEFAULT_ACCURACY_BUDGET = _shared_atif._DEFAULT_ACCURACY_BUDGET
_DEFAULT_GOAL_ACCURACY_BUDGET = _shared_atif._DEFAULT_GOAL_ACCURACY_BUDGET
_BUNDLE_BUDGETS = _shared_atif._BUNDLE_BUDGETS
_BUNDLE_ACCURACY_MAX_OBS = _shared_atif._BUNDLE_ACCURACY_MAX_OBS
_BUNDLE_GOAL_MAX_OBS = _shared_atif._BUNDLE_GOAL_MAX_OBS
_accuracy_budget = _shared_atif._accuracy_budget
_goal_accuracy_budget = _shared_atif._goal_accuracy_budget
_bundle_budgets = _shared_atif._bundle_budgets
_clip = _shared_atif._clip
_late_observation_excerpts = _shared_atif._late_observation_excerpts
_assemble = _shared_atif._assemble
_BACKTICK_TOKEN_RE = _shared_atif._BACKTICK_TOKEN_RE
_VERIFIED_FACTS_MAX = _shared_atif._VERIFIED_FACTS_MAX
_VERIFIED_FACT_LINE_MAX = _shared_atif._VERIFIED_FACT_LINE_MAX
build_verified_facts = _shared_atif.build_verified_facts
_build_verified_facts_section = _shared_atif._build_verified_facts_section
build_metric_evidence_bundles = _shared_atif.build_metric_evidence_bundles


def _slice_with_middle_marker(text, budget, marker, *, max_head=None, fallback_tail=False):
    """Compact *text* to fit *budget* by replacing the middle with *marker*."""
    if budget <= 0 or not text:
        return ""
    if len(text) <= budget:
        return text
    if budget <= len(marker) + 2:
        return text[-budget:] if fallback_tail else text[:budget]
    avail = budget - len(marker)
    head = max(1, avail // 2 if max_head is None else min(max_head, avail // 2))
    tail = max(1, avail - head)
    return f"{text[:head]}{marker}{text[-tail:]}"


def _compact_behavior_conversation(conversation_text, limit=None):
    """Keep both setup context and late outcome evidence in behavior prompts."""
    if limit is None:
        limit = _behavior_check_budget()
    if len(conversation_text) <= limit:
        return conversation_text
    marker = "\n...[middle truncated for behavior check]...\n"
    if limit <= len(marker) + 1:
        return conversation_text[:limit]
    final_limit = _behavior_final_response_limit()
    final_header = f"{_SECTION_FINAL_RESPONSE}\n"
    final_idx = -1
    if conversation_text.startswith(final_header):
        final_idx = 0
    else:
        pos = conversation_text.find(f"\n\n{final_header}")
        if pos != -1:
            final_idx = pos + 2

    if final_idx != -1:
        final_end = len(conversation_text)
        for next_hdr in (f"\n\n{_SECTION_USER_REQUEST}\n", f"\n\n{_SECTION_COMPACT_TOOL_HISTORY}\n"):
            pos = conversation_text.find(next_hdr, final_idx)
            if pos != -1 and pos < final_end:
                final_end = pos
        prefix = conversation_text[:final_idx]
        final_sec = conversation_text[final_idx:final_end]
        suffix = conversation_text[final_end:]

        reserved_other = min(1600, max(0, limit - final_limit), limit // 2)
        max_final = min(final_limit, max(1, limit - reserved_other))
        if len(final_sec) > max_final:
            final_body = final_sec[len(final_header) :]
            body_limit = max(1, max_final - len(final_header))
            final_sec = f"{final_header}{_truncate_for_behavior(final_body, body_limit)}"[:max_final]

        rem = limit - len(final_sec)
        if rem <= 0:
            return final_sec[:limit]
        if len(prefix) + len(suffix) <= rem:
            return f"{prefix}{final_sec}{suffix}"
        if not suffix:
            pre_comp = _slice_with_middle_marker(prefix, rem, marker)
            return f"{pre_comp}{final_sec}"[:limit]
        if len(prefix) <= rem // 2:
            suf_comp = _slice_with_middle_marker(suffix, rem - len(prefix), marker, max_head=800, fallback_tail=True)
            return f"{prefix}{final_sec}{suf_comp}"[:limit]
        pre_budget = max(1, min(len(prefix), rem // 2))
        suf_budget = max(0, rem - pre_budget)
        pre_comp = _slice_with_middle_marker(prefix, pre_budget, marker)
        suf_comp = _slice_with_middle_marker(suffix, suf_budget, marker, max_head=800, fallback_tail=True)
        return f"{pre_comp}{final_sec}{suf_comp}"[:limit]

    available = limit - len(marker)
    reserved_head = min(1600, available // 2)
    tail = max(1, available // 3, min(final_limit, max(1, available - max(1, reserved_head))))
    head = max(1, available - tail)
    return f"{conversation_text[:head]}{marker}{conversation_text[-tail:]}"


# ── Public Provider Caller ───────────────────────────────────────────────────


def _dedupe_models(models):
    seen = set()
    result = []
    for model in models:
        model = str(model or "").strip()
        if model and model not in seen:
            seen.add(model)
            result.append(model)
    return result


def _fallback_models(primary_model):
    env_fallbacks = [
        item.strip() for item in os.environ.get("LLM_JUDGE_FALLBACK_MODELS", "").split(",") if item.strip()
    ]
    return _dedupe_models([primary_model, *env_fallbacks])


def _resolve_url(provider):
    if provider == "nv_build":
        url = os.environ.get("SKILL_EVAL_LLM_BASE_URL") or NVIDIA_BUILD_CHAT_URL
        return _validate_http_url(url)
    base_url = os.environ.get("SKILL_EVAL_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    url = base_url.rstrip("/") + "/chat/completions" if base_url else OPENAI_CHAT_URL
    return _validate_http_url(url)


def _validate_http_url(url):
    """Allow explicit public provider endpoints, not local-file schemes."""
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Provider base URL must be an absolute HTTP or HTTPS URL")
    return url


def _configured_secret_values(extra_secret_values=()):
    values = {
        value
        for name in _CREDENTIAL_ENV_VARS
        if (value := os.environ.get(name, "")) and len(value) >= _MIN_EXACT_SECRET_LENGTH
    }
    for value in extra_secret_values:
        text = str(value) if value else ""
        if len(text) >= _MIN_EXACT_SECRET_LENGTH:
            values.add(text)
    return sorted(values, key=len, reverse=True)


def _redact_configured_credentials(text, extra_secret_values=()):
    redacted = str(text)
    for secret in _configured_secret_values(extra_secret_values):
        redacted = redacted.replace(secret, _ERROR_REDACTION_MARKER)
    return redacted


def _judge_error(error_reason, **metadata):
    """Return a bounded, redacted result that cannot be mistaken for a judged zero."""
    safe_reason = _redact_configured_credentials(error_reason).strip() or "LLM judge failed"
    if len(safe_reason) > _JUDGE_ERROR_REASON_LIMIT:
        safe_reason = safe_reason[: _JUDGE_ERROR_REASON_LIMIT - 3] + "..."
    return {**metadata, "score": None, "status": "error", "reason": safe_reason}


def _bounded_judge_text(value):
    """Normalize trusted-shape model text before it reaches artifacts and reports."""
    text = _redact_configured_credentials(value).strip() if isinstance(value, str) else ""
    if len(text) > _JUDGE_TEXT_LIMIT:
        text = text[: _JUDGE_TEXT_LIMIT - 3] + "..."
    return text


def _finite_score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, int):
        if value <= 0:
            return 0.0
        return 1.0
    score = float(value)
    if not math.isfinite(score):
        return None
    return max(0.0, min(1.0, score))


def _sanitize_error_value(value, extra_secret_values=()):
    secrets = _configured_secret_values(extra_secret_values)

    def sanitize(item):
        if isinstance(item, str):
            redacted = item
            for secret in secrets:
                redacted = redacted.replace(secret, _ERROR_REDACTION_MARKER)
            return redacted
        if isinstance(item, dict):
            return {sanitize(key): sanitize(nested) for key, nested in item.items()}
        if isinstance(item, list):
            return [sanitize(nested) for nested in item]
        if isinstance(item, tuple):
            return tuple(sanitize(nested) for nested in item)
        return item

    return sanitize(value)


def _format_http_error_with_fallback(error):
    try:
        body = error.read().decode("utf-8", "replace").strip()
    except Exception:
        body = ""
    raw_detail = f"HTTP {error.code}: {error.reason}"
    safe_detail = raw_detail
    if body:
        raw_detail = f"{raw_detail} - {body}"
        safe_detail = f"{safe_detail} - {_redact_configured_credentials(body)[:500]}"
    return _redact_configured_credentials(safe_detail), _should_try_fallback(raw_detail)


def _format_http_error(error):
    return _format_http_error_with_fallback(error)[0]


def _should_try_fallback(error):
    text = error.lower()
    return (
        "key_model_access_denied" in text
        or "not allowed to access model" in text
        or "invalid model" in text
        or "model not found" in text
    )


# Model capability helpers. Keep in sync with skillevaluator.provider_config (drift test).


def _model_leaf(model):
    leaf = str(model or "").strip().casefold().rsplit("/", 1)[-1]
    leaf = re.sub(r"^(?:[a-z]+-)+?(?=(?:claude|gpt)-)", "", leaf, count=1)
    return re.sub(r"^(?:(?:[a-z]{2}|global)\.)?anthropic\.", "", leaf, count=1)


def _claude_version(model):
    match = re.fullmatch(
        r"claude-[a-z][a-z-]*-(?P<major>\d+)"
        r"(?:-(?P<minor>\d{1,2}))?"
        r"(?:-(?:\d{8}|latest))?"
        r"(?:-v\d+)?(?::\d+)?",
        _model_leaf(model),
    )
    if match is None:
        return None
    return int(match.group("major")), int(match.group("minor") or 0)


def _is_openai_reasoning_model(model):
    return _model_leaf(model).startswith(("gpt-5", "gpt-6"))


def _is_claude_5_5_or_later(model):
    version = _claude_version(model)
    return version is not None and version >= (5, 5)


def _supports_custom_temperature(model):
    if _is_openai_reasoning_model(model) or _model_leaf(model) == "claude-mythos-preview":
        return False
    version = _claude_version(model)
    return version is None or version < (4, 7)


def _completion_token_limit(model, max_tokens):
    return max(max_tokens, REASONING_MAX_COMPLETION_TOKENS) if _is_openai_reasoning_model(model) else max_tokens


def _effective_reasoning_effort(provider, model, effort):
    if provider == "anthropic":
        supported = _is_claude_5_5_or_later(model)
    else:
        supported = provider != "bedrock" and _model_leaf(model).startswith("gpt-6")
    return effort if supported else None


def _is_native_openai_chat_url(provider, request_url):
    if str(provider or "").strip().casefold() != "openai":
        return False

    raw_url = str(request_url or "")
    if raw_url != raw_url.strip() or any(ord(character) < 0x20 or ord(character) == 0x7F for character in raw_url):
        return False
    try:
        parsed = urlparse(raw_url)
        port = parsed.port
    except ValueError:
        return False

    return (
        parsed.scheme.casefold() == "https"
        and parsed.hostname is not None
        and parsed.hostname.casefold() == "api.openai.com"
        and parsed.netloc.casefold() in {"api.openai.com", "api.openai.com:443"}
        and port in {None, 443}
        and parsed.path in {"/v1/chat/completions", "/v1/chat/completions/"}
        and parsed.username is None
        and parsed.password is None
        and not parsed.params
        and not parsed.query
        and not parsed.fragment
        and ";" not in raw_url
        and "?" not in raw_url
        and "#" not in raw_url
    )


def _is_vertex_openapi_url(request_url):
    """Return whether a request URL targets a Vertex AI OpenAPI endpoint."""
    if not isinstance(request_url, str) or not request_url.strip():
        return False
    try:
        parsed = urlsplit(request_url.strip())
    except Exception:
        return False
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    if not re.fullmatch(r"(?:[a-z0-9]+-[a-z0-9]+(?:-[a-z0-9]+)?-)?aiplatform\.googleapis\.com", host):
        return False
    path = parsed.path.rstrip("/")
    return "/endpoints/openapi" in path


def _get_vertex_access_token(timeout_seconds=10.0):
    """Acquire or refresh a Google Cloud access token in-process for Vertex OpenAPI grading.

    Attempts discovery via:
    1. google.auth.default() (standard ADC library if installed)
    2. GKE / GCE metadata server at http://169.254.169.254 (zero-dependency container ADC)
    3. gcloud auth print-access-token (subprocess CLI fallback)
    """
    # 1. google.auth
    try:
        import google.auth
        import google.auth.transport.requests

        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
        if getattr(credentials, "token", None):
            return str(credentials.token)
    except Exception:
        pass

    # 2. GKE / GCE metadata server
    try:
        metadata_url = "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token"
        req = urllib.request.Request(
            metadata_url,
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=min(timeout_seconds, 5.0)) as resp:  # nosec B310
            data = json.loads(resp.read().decode("utf-8"))
            token = data.get("access_token")
            if token and isinstance(token, str):
                return token.strip()
    except Exception:
        pass

    # 3. gcloud CLI fallback
    import shutil
    import subprocess

    gcloud_path = shutil.which("gcloud")
    if gcloud_path:
        for args in (
            [gcloud_path, "auth", "application-default", "print-access-token"],
            [gcloud_path, "auth", "print-access-token"],
        ):
            try:
                proc = subprocess.run(
                    args,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    return proc.stdout.strip()
            except Exception:
                pass

    return None


def _build_openai_response_format(schema, schema_name="judge_response"):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": schema,
        },
    }


def _build_anthropic_output_config(schema):
    return {
        "format": {
            "type": "json_schema",
            "schema": schema,
        }
    }


def _chat_completion_payload(
    model,
    prompt,
    max_tokens,
    temperature,
    provider=None,
    request_url=None,
    response_schema=None,
    schema_name="judge_response",
):
    resolved_provider = _public_provider() if provider is None else provider
    resolved_request_url = _resolve_url(resolved_provider) if request_url is None else request_url
    token_key = (
        "max_completion_tokens"
        if _is_openai_reasoning_model(model) and _is_native_openai_chat_url(resolved_provider, resolved_request_url)
        else "max_tokens"
    )
    payload = {
        "model": model,
        token_key: _completion_token_limit(model, max_tokens),
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if temperature is not None and _supports_custom_temperature(model):
        payload["temperature"] = temperature
    if effort := _effective_reasoning_effort(resolved_provider, model, JUDGE_REASONING_EFFORT):
        payload["reasoning_effort"] = effort
    if response_schema is not None:
        payload["response_format"] = _build_openai_response_format(response_schema, schema_name)
    return payload


def _public_provider():
    configured = os.environ.get("SKILL_EVAL_LLM_PROVIDER", "").strip().lower()
    if configured:
        return configured
    providers = _configured_public_providers()
    return providers[0] if len(providers) == 1 else ""


def _configured_public_providers():
    providers = []
    if os.environ.get("OPENAI_API_KEY"):
        providers.append("openai")
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers.append("anthropic")
    if os.environ.get("NVIDIA_API_KEY"):
        providers.append("nv_build")
    if os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE"):
        providers.append("bedrock")
    return providers


def _public_provider_error():
    providers = _configured_public_providers()
    if len(providers) > 1:
        return "Set SKILL_EVAL_LLM_PROVIDER because multiple provider credentials are configured"
    return "Configure SKILL_EVAL_LLM_PROVIDER and a public provider credential"


def _canonical_anthropic_authority(netloc, hostname):
    if netloc.startswith("["):
        closing_bracket = netloc.find("]")
        if closing_bracket < 0:
            return None
        literal = netloc[1:closing_bracket]
        suffix = netloc[closing_bracket + 1 :]
        if literal.casefold() != hostname.casefold() or (
            suffix and (not suffix.startswith(":") or not suffix[1:].isascii() or not suffix[1:].isdigit())
        ):
            return None

        address = literal
        zone = ""
        if "%" in literal:
            address, separator, zone = literal.partition("%25")
            if not separator or "%" in address or "%" in zone or not _ANTHROPIC_IPV6_ZONE_RE.fullmatch(zone):
                return None
        try:
            ipaddress.IPv6Address(address)
        except ValueError:
            return None
        return f"[{address}{'%25' + zone if zone else ''}]{suffix}"

    if "%" in netloc or "[" in netloc or "]" in netloc:
        return None
    host = netloc
    suffix = ""
    if ":" in netloc:
        host, port = netloc.rsplit(":", maxsplit=1)
        if ":" in host or not port.isascii() or not port.isdigit():
            return None
        suffix = f":{port}"
    if host.casefold() != hostname.casefold():
        return None

    if "." in hostname and all(character in "0123456789." for character in hostname):
        try:
            ipaddress.IPv4Address(hostname)
        except ValueError:
            return None
        return f"{host}{suffix}"

    trailing_dot = host.endswith(".")
    dns_name = host.removesuffix(".")
    if not dns_name:
        return None
    if dns_name.isascii() and "_" in dns_name:
        canonical_name = dns_name.lower()
        label_pattern = _ANTHROPIC_INTERNAL_LABEL_RE
    else:
        try:
            canonical_name = idna.encode(dns_name.lower()).decode("ascii")
        except idna.IDNAError:
            return None
        label_pattern = _ANTHROPIC_DNS_LABEL_RE
    if len(canonical_name) > 253 or not all(label_pattern.fullmatch(label) for label in canonical_name.split(".")):
        return None
    return f"{canonical_name}{'.' if trailing_dot else ''}{suffix}"


def _canonical_anthropic_path(path):
    canonical = []
    index = 0
    while index < len(path):
        character = path[index]
        if character != "%":
            canonical.append(character)
            index += 1
            continue

        if index + 2 >= len(path) or path[index + 1] not in _HEX_DIGITS or path[index + 2] not in _HEX_DIGITS:
            return None
        octet = int(path[index + 1 : index + 3], 16)
        if octet in {0x2F, 0x5C, 0x7F} or octet < 0x20:
            return None
        if octet in _UNRESERVED_BYTES:
            canonical.append(chr(octet))
        else:
            canonical.append(f"%{octet:02X}")
        index += 3

    canonical_path = "".join(canonical)
    if "//" in canonical_path.rstrip("/"):
        return None
    decoded_octets = unquote_to_bytes(canonical_path)
    # A decoded percent is safe as data unless it opens a second escape layer.
    if any(
        decoded_octets[index] == 0x25
        and index + 2 < len(decoded_octets)
        and decoded_octets[index + 1] in _HEX_DIGIT_BYTES
        and decoded_octets[index + 2] in _HEX_DIGIT_BYTES
        for index in range(len(decoded_octets))
    ):
        return None
    decoded_path = decoded_octets.decode("utf-8", errors="replace")
    if any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in decoded_path):
        return None
    if any(segment in {".", ".."} for segment in decoded_path.split("/")):
        return None
    return quote(canonical_path, safe=_ANTHROPIC_PATH_SAFE)


def _normalize_anthropic_base_url(value, variable):
    error = (
        f"{variable} must be an absolute HTTP or HTTPS URL representing an API root without credentials, query, fragment, "
        "whitespace, control characters, backslashes, an invalid authority, or a /v1/messages endpoint."
    )
    if "\\" in value or any(
        character.isspace() or unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in value
    ):
        raise ValueError(error)
    if "?" in value or "#" in value:
        raise ValueError(error)

    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError:
        raise ValueError(error) from None
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.netloc
        or hostname is None
        or parsed.netloc.endswith(":")
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(error)

    authority = _canonical_anthropic_authority(parsed.netloc, hostname)
    path = _canonical_anthropic_path(parsed.path)
    if authority is None or path is None:
        raise ValueError(error)

    path = path.rstrip("/")
    if path.endswith("/v1/messages"):
        raise ValueError(error)
    if path.endswith("/v1"):
        path = path.removesuffix("/v1")
    return parsed._replace(netloc=authority, path=path, query="", fragment="").geturl()


def _anthropic_url():
    for variable in ("SKILL_EVAL_LLM_BASE_URL", "ANTHROPIC_BASE_URL"):
        if base_url := os.environ.get(variable):
            root = _normalize_anthropic_base_url(base_url, variable)
            url = root + "/v1/messages"
            break
    else:
        url = "https://api.anthropic.com/v1/messages"
    return _validate_http_url(url)


_RETRIABLE_HTTP_CODES = frozenset({429, 500, 502, 503, 504})
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BASE_DELAY = 1.0
_DEFAULT_MAX_DELAY = 30.0
# Three required judges run sequentially under the managed 600-second Harbor
# verifier timeout. Reserve one minute for deterministic checks and artifacts.
_JUDGE_WALL_TIME_BUDGET_SEC = 180.0
_ACTIVE_JUDGE_DEADLINE: ContextVar[float | None] = ContextVar("active_judge_deadline", default=None)
_ACTIVE_JUDGE_CALLS: ContextVar[list[dict[str, Any]] | None] = ContextVar("active_judge_calls", default=None)


class EvalRetryConfig(NamedTuple):
    """Represent bounded retry and backoff settings for direct verifier LLM calls."""

    max_retries: int
    base_delay: float
    max_delay: float


class SchemaTargetKey(NamedTuple):
    """Identify a provider endpoint and model for structured output schema memoization."""

    provider: str
    base_url: str
    model: str


def _resolve_judge_wall_time_budget():
    """Resolve the per-judge wall-time budget in seconds from the environment."""
    raw = str(os.environ.get("SKILL_EVAL_LLM_JUDGE_BUDGET_SEC", "")).strip()
    if raw:
        try:
            val = float(raw)
            if math.isfinite(val) and val > 0.0:
                return val
        except ValueError:
            pass
    return _JUDGE_WALL_TIME_BUDGET_SEC


def _remaining_judge_timeout(timeout):
    """Bound one provider request by the remaining time for its judge."""
    deadline = _ACTIVE_JUDGE_DEADLINE.get()
    if deadline is None:
        return timeout
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("LLM judge time budget exhausted")
    return min(timeout, remaining)


def _parse_retry_after(header_value, fallback_delay):
    """Parse a Retry-After header as seconds or HTTP date, falling back to default."""
    if not header_value:
        return fallback_delay
    clean_val = str(header_value).strip()
    try:
        return max(0.0, float(clean_val))
    except ValueError:
        pass
    try:
        from datetime import UTC, datetime
        from email.utils import parsedate_to_datetime

        target = parsedate_to_datetime(clean_val)
        if target.tzinfo is None:
            target = target.replace(tzinfo=UTC)
        now = datetime.now(UTC)
        return max(0.0, (target - now).total_seconds())
    except Exception:
        return fallback_delay


def _calculate_jitter_delay(attempt, base_delay=1.0, max_delay=30.0):
    """Calculate exponential backoff with full jitter."""
    calculated = min(max_delay, base_delay * (2.0**attempt))
    return random.uniform(0.0, calculated)


def _resolve_eval_retry_config():
    """Resolve retry and backoff limits from environment variables with safe defaults."""

    def _read_int(name, default):
        """Read a non-negative integer from the environment variable or return default."""
        raw = str(os.environ.get(name, "")).strip()
        if raw:
            try:
                val = int(raw)
                return val if val >= 0 else default
            except ValueError:
                return default
        return default

    def _read_float(name, default):
        """Read a non-negative float from the environment variable or return default."""
        raw = str(os.environ.get(name, "")).strip()
        if raw:
            try:
                val = float(raw)
                return val if math.isfinite(val) and val >= 0.0 else default
            except ValueError:
                return default
        return default

    max_retries = _read_int("SKILL_EVAL_LLM_MAX_RETRIES", _DEFAULT_MAX_RETRIES)
    base_delay = _read_float("SKILL_EVAL_LLM_RETRY_BASE_DELAY", _DEFAULT_BASE_DELAY)
    raw_max_delay = _read_float("SKILL_EVAL_LLM_RETRY_MAX_DELAY", _DEFAULT_MAX_DELAY)
    return EvalRetryConfig(max_retries=max_retries, base_delay=base_delay, max_delay=max(base_delay, raw_max_delay))


def _compute_bounded_retry_delay(retry_after_str, *, attempt, base_delay, max_delay, error):
    """Compute a bounded retry sleep duration and verify the judge deadline allows it."""
    if retry_after_str is not None:
        parsed = _parse_retry_after(retry_after_str, fallback_delay=base_delay)
        if parsed > max_delay:
            raise error
        delay = parsed + random.uniform(0.1, 0.5)
    else:
        delay = _calculate_jitter_delay(attempt, base_delay=base_delay, max_delay=max_delay)

    sleep_duration = min(delay, max_delay)
    deadline = _ACTIVE_JUDGE_DEADLINE.get()
    if deadline is not None and time.monotonic() + sleep_duration >= deadline:
        raise TimeoutError("LLM judge time budget exhausted before retry") from error
    return sleep_duration


def _urlopen_with_retry(request, timeout=90):
    """Open a URL request with exponential backoff and full jitter on transient failures."""
    retry_config = _resolve_eval_retry_config()
    attempt = 0
    while True:
        request_timeout = _remaining_judge_timeout(timeout)
        try:
            with urllib.request.urlopen(request, timeout=request_timeout) as response:  # nosec B310
                return response.read()
        except Exception as error:
            is_http = isinstance(error, urllib.error.HTTPError)
            is_network = isinstance(error, (urllib.error.URLError, ConnectionError, OSError))
            if (
                attempt >= retry_config.max_retries
                or (not is_http and not is_network)
                or (is_http and error.code not in _RETRIABLE_HTTP_CODES)
            ):
                raise

            retry_after_str = None
            if is_http and error.headers:
                retry_after_str = error.headers.get("retry-after") or error.headers.get("Retry-After")

            sleep_duration = _compute_bounded_retry_delay(
                retry_after_str,
                attempt=attempt,
                base_delay=retry_config.base_delay,
                max_delay=retry_config.max_delay,
                error=error,
            )
            status_label = f"HTTP {error.code}" if is_http else type(error).__name__
            logger.warning(
                "LLM judge transient error (%s). Retrying in %.2fs (attempt %d/%d)...",
                status_label,
                sleep_duration,
                attempt + 1,
                retry_config.max_retries,
            )
            if is_http:
                error.close()
            time.sleep(sleep_duration)
            attempt += 1


_UNSUPPORTED_REASON_INDICATORS = (
    "unsupported",
    "not supported",
    "extra input",
    "extra inputs",
    "unknown parameter",
    "unknown field",
    "unknown argument",
    "unrecognized request argument",
    "unrecognized parameter",
    "unexpected keyword argument",
    "unexpected argument",
    "invalid parameter",
    "invalid argument",
    "not permitted",
    "not allowed",
    "disallowed",
    "unavailable",
)

# Gateways can turn a schema into a forced tool call, so a tool_choice rejection is a schema rejection too.
_SCHEMA_OPTION_PATTERN = (
    r"(?:response_format|response format|output_config|json_schema|structured[_ ]outputs?|tool_choice)"
)
_SCHEMA_REJECTION_REASON = (
    r"(?:unsupported|not supported|unavailable|not permitted|not allowed|disallowed|"
    r"unknown (?:parameter|field|argument)|unrecognized (?:request argument|parameter)|"
    r"unexpected (?:keyword )?argument|extra inputs?(?: are not permitted)?)"
)
_SCHEMA_REJECTION_AFTER_OPTION = re.compile(
    rf"\b{_SCHEMA_OPTION_PATTERN}\b(?:\.[a-z0-9_]+)*"
    rf"(?:\s+type)?"
    rf"(?:\s+of\s+type\s+['\"]?[a-z0-9_]+['\"]?)?"
    rf"\s*(?:(?:is|are|was|were)\s+(?:an?\s+)?|:\s*)?"
    rf"{_SCHEMA_REJECTION_REASON}\b",
    re.IGNORECASE,
)
_SCHEMA_REJECTION_BEFORE_OPTION = re.compile(
    rf"\b(?:unsupported|not supported|unavailable|extra inputs?(?: are not permitted)?|"
    rf"unknown (?:parameter|field|argument)|unrecognized (?:request argument|parameter)|"
    rf"unexpected (?:keyword argument|argument)|invalid (?:parameter|argument)|"
    rf"not permitted|not allowed|disallowed)\b"
    rf"(?:\s+supplied)?[\s:'\"\[\]{{}}(),-]{{0,32}}\b{_SCHEMA_OPTION_PATTERN}\b",
    re.IGNORECASE,
)


def _message_rejects_schema_option(text, param=None):
    """Match a rejection of the schema option itself, not unrelated error text."""
    if param:
        if not re.search(rf"\b{_SCHEMA_OPTION_PATTERN}\b", param, re.IGNORECASE):
            return False
        return any(indicator in text.lower() for indicator in _UNSUPPORTED_REASON_INDICATORS)
    return bool(_SCHEMA_REJECTION_AFTER_OPTION.search(text) or _SCHEMA_REJECTION_BEFORE_OPTION.search(text))


def _is_schema_unsupported_http_error(error):
    """Determine whether an HTTP error indicates structured output schema is unsupported."""
    if getattr(error, "code", None) not in {400, 422}:
        return False
    body_text = ""
    try:
        body_bytes = error.read()
        error.fp = io.BytesIO(body_bytes)
        body_text = body_bytes.decode("utf-8", "replace")
    except Exception:
        pass
    error_param = None
    try:
        body = json.loads(body_text)
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            param = body["error"].get("param")
            if isinstance(param, str):
                error_param = param
    except (TypeError, ValueError):
        pass
    text = f"{error} {getattr(error, 'reason', '')} {body_text}"
    return _message_rejects_schema_option(text, error_param)


_SCHEMA_UNSUPPORTED_TARGETS: set[SchemaTargetKey] = set()


class _JudgeRefusedError(RuntimeError):
    """The judge model declined the request (a safety-classifier refusal or content filter)."""


def _urlopen_with_schema_fallback(build_request, *, target_key, use_schema, timeout=90):
    """Open URL with retry, falling back to prompt-only on confirmed schema capability errors."""
    normalized_key = target_key if isinstance(target_key, SchemaTargetKey) else SchemaTargetKey(*target_key)
    try:
        return _urlopen_with_retry(build_request(use_schema), timeout=timeout)
    except urllib.error.HTTPError as error:
        if use_schema and _is_schema_unsupported_http_error(error):
            error.close()
            logger.warning(
                "Structured output schema unsupported by provider=%s model=%s; "
                "downgrading to prompt-only JSON and memoizing target.",
                normalized_key.provider,
                normalized_key.model,
            )
            response = _urlopen_with_retry(build_request(False), timeout=timeout)
            _SCHEMA_UNSUPPORTED_TARGETS.add(normalized_key)
            return response
        raise


def _call_anthropic(prompt, model, max_tokens, temperature, response_schema=None, provenance=None):
    """Call the Messages API; ``provenance["model"]`` becomes the model that served the reply."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None, "ANTHROPIC_API_KEY is required for the anthropic provider"
    target_url = _anthropic_url()
    target_key = SchemaTargetKey(provider="anthropic", base_url=target_url, model=model)
    use_schema = response_schema is not None and target_key not in _SCHEMA_UNSUPPORTED_TARGETS
    effort = _effective_reasoning_effort("anthropic", model, JUDGE_REASONING_EFFORT)
    # Server-side fallback (beta, native Claude API only): a safety-classifier
    # refusal is retried on the fallback model within the same call.
    server_side_fallback = _is_claude_5_5_or_later(model) and urlparse(target_url).hostname == "api.anthropic.com"

    def _build_request(include_schema):
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if temperature is not None and _supports_custom_temperature(model):
            payload["temperature"] = temperature
        output_config = _build_anthropic_output_config(response_schema) if include_schema else {}
        if effort is not None:
            output_config["effort"] = effort
        if output_config:
            payload["output_config"] = output_config
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        if server_side_fallback:
            payload["fallbacks"] = [{"model": ANTHROPIC_REFUSAL_FALLBACK_MODEL}]
            headers["anthropic-beta"] = ANTHROPIC_SERVER_SIDE_FALLBACK_BETA
        return urllib.request.Request(target_url, data=json.dumps(payload).encode(), headers=headers)

    # _anthropic_url() validates the configured base URL before this request.
    raw_response = _urlopen_with_schema_fallback(
        _build_request,
        target_key=target_key,
        use_schema=use_schema,
        timeout=90,
    )
    body = json.loads(raw_response)
    if body.get("stop_reason") == "refusal":
        raise _JudgeRefusedError("judge model declined the request (stop_reason=refusal)")
    blocks = [block for block in body.get("content", []) if isinstance(block, dict)]
    for block in blocks:
        served = block.get("to") if block.get("type") == "fallback" else None
        if provenance is not None and isinstance(served, dict) and isinstance(served.get("model"), str):
            provenance["model"] = served["model"]
    content = "".join(str(block.get("text", "")) for block in blocks if block.get("type") == "text")
    return content.strip(), None


_RETRIABLE_BEDROCK_ERROR_CODES = frozenset(
    {
        "ThrottlingException",
        "Throttling",
        "TooManyRequestsException",
        "ServiceUnavailableException",
        "ServiceUnavailable",
        "InternalServerException",
        "InternalServerError",
        "InternalFailure",
        "ModelTimeoutException",
        "RequestTimeout",
        "RequestTimeoutException",
    }
)
_RETRIABLE_BOTOCORE_EXCEPTION_NAMES = frozenset(
    {
        "EndpointConnectionError",
        "ConnectionClosedError",
        "ReadTimeoutError",
        "ConnectTimeoutError",
    }
)


def _classify_bedrock_retry_error(error):
    """Return (is_retriable, status_label, retry_after_str) for a Bedrock Converse exception."""
    if isinstance(error, TimeoutError) and "LLM judge time budget exhausted" in str(error):
        return False, type(error).__name__, None
    if isinstance(error, (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError)):
        return False, type(error).__name__, None

    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        response = {}
    metadata = response.get("ResponseMetadata")
    if not isinstance(metadata, dict):
        metadata = {}
    http_status = metadata.get("HTTPStatusCode")
    if not isinstance(http_status, int) or isinstance(http_status, bool):
        http_status = None

    err_info = response.get("Error")
    if not isinstance(err_info, dict):
        err_info = {}
    error_code = str(err_info.get("Code") or "").strip()

    headers = metadata.get("HTTPHeaders")
    retry_after_str = None
    if isinstance(headers, dict):
        for k, v in headers.items():
            if isinstance(k, str) and k.lower() == "retry-after" and v is not None:
                retry_after_str = str(v)
                break

    if http_status is not None or error_code:
        is_retriable = (
            http_status in _RETRIABLE_HTTP_CODES or http_status == 408 or error_code in _RETRIABLE_BEDROCK_ERROR_CODES
        )
        status_label = f"HTTP {http_status}" if http_status is not None else error_code
        return is_retriable, status_label, retry_after_str

    type_name = type(error).__name__
    is_network = (
        isinstance(error, (ConnectionError, TimeoutError, OSError)) or type_name in _RETRIABLE_BOTOCORE_EXCEPTION_NAMES
    )
    return is_network, type_name, None


def _bedrock_refusal_fallback_model(model):
    """Return Opus 4.8 in the same Bedrock geography as a Claude 5.5 or later model, else None."""
    prefix = re.match(r"(?:(?:[a-z]{2}|global)\.)?anthropic\.", str(model or ""))
    if prefix is None or not _is_claude_5_5_or_later(model):
        return None
    return prefix.group(0) + ANTHROPIC_REFUSAL_FALLBACK_MODEL


def _call_bedrock(prompt, model, max_tokens, temperature, timeout=90, provenance=None):
    """Call Bedrock Converse; ``provenance["model"]`` becomes the model that served the reply."""
    try:
        import boto3
    except ImportError:
        return None, "boto3 is required for the bedrock provider"
    BotoConfig = None
    with contextlib.suppress(ImportError):
        from botocore.config import Config as BotoConfig
    try:
        retry_config = _resolve_eval_retry_config()
        region_name = os.environ.get("AWS_REGION", "us-west-2")
        initial_timeout = _remaining_judge_timeout(timeout)
        client_kwargs = {"region_name": region_name}
        if BotoConfig is not None:
            client_kwargs["config"] = BotoConfig(
                connect_timeout=initial_timeout,
                read_timeout=initial_timeout,
                retries={"max_attempts": 0, "mode": "standard"},
            )
        try:
            client = boto3.client("bedrock-runtime", **client_kwargs)
        except TypeError:
            client = boto3.client("bedrock-runtime", region_name=region_name)

        inference_config = {"maxTokens": max_tokens}
        if temperature is not None and _supports_custom_temperature(model):
            inference_config["temperature"] = temperature

        def _converse(model_id):
            attempt = 0
            while True:
                request_timeout = _remaining_judge_timeout(timeout)
                endpoint = getattr(client, "_endpoint", None)
                if endpoint is not None and hasattr(endpoint, "timeout"):
                    endpoint.timeout = request_timeout
                try:
                    return client.converse(
                        modelId=model_id,
                        messages=[{"role": "user", "content": [{"text": prompt}]}],
                        inferenceConfig=inference_config,
                    )
                except Exception as error:
                    is_retriable, status_label, retry_after_str = _classify_bedrock_retry_error(error)
                    if attempt >= retry_config.max_retries or not is_retriable:
                        raise

                    sleep_duration = _compute_bounded_retry_delay(
                        retry_after_str,
                        attempt=attempt,
                        base_delay=retry_config.base_delay,
                        max_delay=retry_config.max_delay,
                        error=error,
                    )
                    logger.warning(
                        "LLM judge transient error (%s). Retrying in %.2fs (attempt %d/%d)...",
                        status_label,
                        sleep_duration,
                        attempt + 1,
                        retry_config.max_retries,
                    )
                    time.sleep(sleep_duration)
                    attempt += 1

        response = _converse(model)
        fallback_model = _bedrock_refusal_fallback_model(model)
        if response.get("stopReason") == "content_filtered" and fallback_model:
            # Bedrock has no server-side fallback, so retry a classifier refusal here.
            response = _converse(fallback_model)
            if provenance is not None:
                provenance["model"] = fallback_model
        if response.get("stopReason") == "content_filtered":
            raise _JudgeRefusedError("judge model declined the request (stopReason=content_filtered)")
        content = "".join(
            str(block.get("text", ""))
            for block in response.get("output", {}).get("message", {}).get("content", [])
            if isinstance(block, dict)
        )
        return content.strip(), None
    except _JudgeRefusedError:
        raise
    except Exception as exc:
        return None, f"Bedrock request failed: {exc}"


def _selected_judge_model(model=None):
    return (
        model
        or os.environ.get("LLM_JUDGE_MODEL")
        or os.environ.get("SKILL_EVAL_JUDGE_MODEL")
        or os.environ.get("SKILL_EVAL_LLM_MODEL")
        or DEFAULT_JUDGE_MODELS.get(_public_provider(), DEFAULT_JUDGE_MODELS["openai"])
    )


def _call_openai_compatible(
    prompt, provider, model, max_tokens, temperature, response_schema=None, schema_name="judge_response"
):
    api_key = (
        os.environ.get("NVIDIA_API_KEY", "")
        if provider == "nv_build"
        else os.environ.get("SKILL_EVAL_LLM_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    )
    if not api_key:
        return None, f"No API key configured for {provider}"
    request_url = _resolve_url(provider)
    target_key = SchemaTargetKey(provider=provider, base_url=request_url, model=model)
    use_schema = response_schema is not None and target_key not in _SCHEMA_UNSUPPORTED_TARGETS

    def _build_request(include_schema, *, _key=api_key):
        payload = _chat_completion_payload(
            model,
            prompt,
            max_tokens,
            temperature,
            provider=provider,
            request_url=request_url,
            response_schema=response_schema if include_schema else None,
            schema_name=schema_name,
        )
        return urllib.request.Request(
            request_url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {_key}"},
        )

    # request_url was validated by _resolve_url() before this request.
    try:
        raw_response = _urlopen_with_schema_fallback(
            _build_request,
            target_key=target_key,
            use_schema=use_schema,
            timeout=90,
        )
    except urllib.error.HTTPError as http_err:
        if (
            http_err.code == 401
            and _is_vertex_openapi_url(request_url)
            and os.environ.get("SKILL_EVAL_LLM_CREDENTIAL_SOURCE", "").strip() == "ADC"
        ):
            logger.info("Vertex AI OpenAPI 401 received; attempting in-process ADC token refresh")
            refreshed_token = _get_vertex_access_token()
            if refreshed_token and refreshed_token != api_key:
                if "OPENAI_API_KEY" in os.environ:
                    os.environ["OPENAI_API_KEY"] = refreshed_token
                if "SKILL_EVAL_LLM_API_KEY" in os.environ:
                    os.environ["SKILL_EVAL_LLM_API_KEY"] = refreshed_token
                raw_response = _urlopen_with_schema_fallback(
                    partial(_build_request, _key=refreshed_token),
                    target_key=target_key,
                    use_schema=use_schema,
                    timeout=90,
                )
            else:
                raise
        else:
            raise
    body = json.loads(raw_response)
    choices = body.get("choices") or [{}]
    first_choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = first_choice.get("message")
    refusal = message.get("refusal") if isinstance(message, dict) else None
    if first_choice.get("finish_reason") == "content_filter" or (isinstance(refusal, str) and refusal):
        raise _JudgeRefusedError(
            f"judge model declined the request (finish_reason={first_choice.get('finish_reason')})"
        )
    content = message.get("content", "") if isinstance(message, dict) else ""
    if content is None:
        content = ""
    return content.strip(), None


def _record_judge_call(provenance, requested_model, sent_model):
    """Record the model that answered for the required judge in progress (see _call_required_judge)."""
    calls = _ACTIVE_JUDGE_CALLS.get()
    if calls is not None:
        calls.append(
            {
                "model": provenance["model"],
                "reasoning_effort": _effective_reasoning_effort(
                    provenance["provider"], sent_model, JUDGE_REASONING_EFFORT
                ),
                "fallback_used": provenance["model"] != requested_model,
            }
        )


def _call_public_llm_with_provenance(
    prompt,
    model=None,
    max_tokens=4096,
    temperature=0.0,
    allow_model_fallback=True,
    response_schema=None,
    schema_name="judge_response",
):
    provider = _public_provider()
    if not provider:
        return None, _public_provider_error(), {}
    requested_model = _selected_judge_model(model)
    models = _fallback_models(requested_model) if allow_model_fallback else [requested_model]
    errors = []
    last_provenance = {"provider": provider, "model": requested_model}
    for candidate_model in models:
        provenance = {"provider": provider, "model": candidate_model}
        last_provenance = provenance
        try:
            if provider == "anthropic":
                content, error = _call_anthropic(
                    prompt,
                    candidate_model,
                    max_tokens,
                    temperature,
                    response_schema=response_schema,
                    provenance=provenance,
                )
            elif provider == "bedrock":
                content, error = _call_bedrock(prompt, candidate_model, max_tokens, temperature, provenance=provenance)
            else:
                content, error = _call_openai_compatible(
                    prompt,
                    provider,
                    candidate_model,
                    max_tokens,
                    temperature,
                    response_schema=response_schema,
                    schema_name=schema_name,
                )
        except _JudgeRefusedError as refusal:
            # A classifier refusal is fallback-eligible like a missing model.
            errors.append(f"{candidate_model}: {refusal}")
            if not allow_model_fallback:
                return None, str(refusal), provenance
            continue
        except urllib.error.HTTPError as http_error:
            detail, should_try_fallback = _format_http_error_with_fallback(http_error)
            errors.append(f"{candidate_model}: {detail}")
            if not allow_model_fallback or not should_try_fallback:
                return None, detail, provenance
            continue
        except Exception as exc:
            detail = f"Public provider call failed for {candidate_model}: {exc}"
            return None, _redact_configured_credentials(detail), provenance
        if error:
            return None, _redact_configured_credentials(error), provenance
        if not content and candidate_model != models[-1]:
            # So is an empty reply, for example reasoning that used up the output budget.
            errors.append(f"{candidate_model}: empty response")
            continue
        if candidate_model != requested_model:
            logger.warning("LLM judge model %s failed; using fallback model %s", requested_model, candidate_model)
        _record_judge_call(provenance, requested_model, candidate_model)
        return content, None, provenance
    detail = "LLM judge model fallback exhausted: " + " | ".join(errors)
    return None, _redact_configured_credentials(detail), last_provenance


def call_public_llm(
    prompt,
    model=None,
    max_tokens=4096,
    temperature=0.0,
    allow_model_fallback=True,
    response_schema=None,
    schema_name="judge_response",
):
    content, error, _provenance = _call_public_llm_with_provenance(
        prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        allow_model_fallback=allow_model_fallback,
        response_schema=response_schema,
        schema_name=schema_name,
    )
    return content, error


_JSON_WHITESPACE = " \t\r\n"
_MAX_JSON_TEXT_CHARS = 100_000
_MAX_JSON_NESTING = 128
_JSON_NUMBER_PREFIX_RE = re.compile(r"-?(?:(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]*)?|(?:0|[1-9][0-9]*)\.)?")


def _balanced_json_container_end(text, start):
    """Return the exclusive end of one bounded structural container."""
    if start >= len(text) or text[start] not in "{[":
        return None
    stack = []
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
            if len(stack) > _MAX_JSON_NESTING:
                return None
        elif ch in "}]":
            expected = "{" if ch == "}" else "["
            if not stack or stack[-1] != expected:
                return None
            stack.pop()
            if not stack:
                return i + 1
    return None


def _reject_duplicate_object_pairs(pairs):
    """Build an object while rejecting ambiguous duplicate members."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object member")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(_value):
    raise ValueError("Non-standard JSON constant")


def _parse_finite_json_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON number overflowed to a non-finite value")
    return parsed


def _json_nesting_within_limit(text):
    """Bound structural nesting without recursively parsing partial JSON."""
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "{[":
            depth += 1
            if depth > _MAX_JSON_NESTING:
                return False
        elif character in "}]" and depth:
            depth -= 1
    return True


def extract_json(text):
    """Extract a JSON payload from LLM response text.

    Tolerates markdown fences and prose around exactly one valid bounded JSON
    container. Multiple complete documents are ambiguous, and an unfinished
    earlier structural segment blocks promotion of a nested object. Top-level
    arrays parse through unchanged; judge callers must dict-check the result
    themselves.
    """
    text = (text or "").strip()
    if not text or len(text) > _MAX_JSON_TEXT_CHARS:
        return None

    documents = []
    index = 0
    while index < len(text):
        if text[index] not in "{[":
            index += 1
            continue
        end = _balanced_json_container_end(text, index)
        if end is None:
            return documents[0] if documents else None
        candidate = text[index:end]
        try:
            parsed = json.loads(
                candidate,
                object_pairs_hook=_reject_duplicate_object_pairs,
                parse_constant=_reject_nonstandard_json_constant,
                parse_float=_parse_finite_json_float,
            )
        except (json.JSONDecodeError, RecursionError, ValueError):
            index = end
            continue
        if isinstance(parsed, (dict, list)):
            documents.append(parsed)
            if len(documents) > 1:
                return None
        index = end
    return documents[0] if documents else None


def _is_json_string_prefix(text):
    """Return whether an unfinished bounded string can be completed as JSON."""
    if not text.startswith('"'):
        return False
    index = 1
    while index < len(text):
        character = text[index]
        if ord(character) < 0x20 or character == '"':
            return False
        if character != "\\":
            index += 1
            continue
        index += 1
        if index >= len(text):
            return True
        escape = text[index]
        if escape == "u":
            for offset in range(1, 5):
                if index + offset >= len(text):
                    return True
                if text[index + offset] not in "0123456789abcdefABCDEF":
                    return False
            index += 5
        elif escape in '"\\/bfnrt':
            index += 1
        else:
            return False
    return True


def _is_json_scalar_prefix(text):
    if not text:
        return True
    if text.startswith('"'):
        return _is_json_string_prefix(text)
    literals = {"t": "true", "f": "false", "n": "null"}
    if text[0] in literals:
        return literals[text[0]].startswith(text)
    if text[0] == "-" or text[0] in "0123456789":
        return _JSON_NUMBER_PREFIX_RE.fullmatch(text) is not None
    return False


def _is_append_only_json_object_prefix(fragment):
    """Validate an unfinished flat result entry using bounded decoder steps."""
    if not fragment or len(fragment) > _MAX_JSON_TEXT_CHARS:
        return False

    decoder = json.JSONDecoder(
        object_pairs_hook=_reject_duplicate_object_pairs,
        parse_constant=_reject_nonstandard_json_constant,
        parse_float=_parse_finite_json_float,
    )

    def _skip_whitespace(index):
        while index < len(fragment) and fragment[index] in _JSON_WHITESPACE:
            index += 1
        return index

    index = _skip_whitespace(0)
    if index >= len(fragment) or fragment[index] != "{":
        return False
    index += 1
    keys = set()
    while True:
        index = _skip_whitespace(index)
        if index >= len(fragment):
            return True
        if fragment[index] == "}":
            return False
        try:
            key, next_index = decoder.raw_decode(fragment, index)
        except (json.JSONDecodeError, RecursionError, ValueError):
            return _is_json_string_prefix(fragment[index:])
        if not isinstance(key, str) or key in keys:
            return False
        keys.add(key)
        index = _skip_whitespace(next_index)
        if index >= len(fragment):
            return True
        if fragment[index] != ":":
            return False
        index = _skip_whitespace(index + 1)
        if index >= len(fragment):
            return True
        value_start = index
        try:
            value, next_index = decoder.raw_decode(fragment, index)
        except (json.JSONDecodeError, RecursionError, ValueError):
            return _is_json_scalar_prefix(fragment[value_start:])
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and next_index < len(fragment)
            and fragment[next_index] in ".eE"
        ):
            return _JSON_NUMBER_PREFIX_RE.fullmatch(fragment[value_start:]) is not None
        index = _skip_whitespace(next_index)
        if index >= len(fragment):
            return True
        if fragment[index] == "}":
            return False
        if fragment[index] != ",":
            return False
        index += 1


def _salvage_behavior_results(text):
    """Recover complete per-behavior entries from a truncated ``results`` array.

    Reasoning judges that hit the output-token cap emit ``{"results": [...`` and
    stop mid-entry (``finish_reason="length"``); every fully-formed ``{...}``
    entry before the cut is still valid JSON and can be scored.
    """
    text = text or ""
    if len(text) > _MAX_JSON_TEXT_CHARS or not _json_nesting_within_limit(text):
        return []
    object_start = text.find("{")
    if object_start == -1:
        return []
    if any(character in "[]{}" for character in text[:object_start]):
        return []

    decoder = json.JSONDecoder(
        object_pairs_hook=_reject_duplicate_object_pairs,
        parse_constant=_reject_nonstandard_json_constant,
        parse_float=_parse_finite_json_float,
    )

    def _skip_whitespace(index):
        while index < len(text) and text[index] in " \t\r\n":
            index += 1
        return index

    # Parse only complete top-level fields preceding ``results``. This rejects
    # nested/unrelated arrays and lets us validate a score emitted before the
    # array without requiring the outer object itself to be complete.
    i = object_start + 1
    array_start = None
    seen_keys = set()
    while i < len(text):
        i = _skip_whitespace(i)
        try:
            key, i = decoder.raw_decode(text, i)
        except (json.JSONDecodeError, RecursionError, ValueError):
            return []
        if not isinstance(key, str):
            return []
        if key in seen_keys:
            return []
        seen_keys.add(key)
        i = _skip_whitespace(i)
        if i >= len(text) or text[i] != ":":
            return []
        i = _skip_whitespace(i + 1)
        if key == "results":
            if i >= len(text) or text[i] != "[":
                return []
            array_start = i
            break
        try:
            value, i = decoder.raw_decode(text, i)
        except (json.JSONDecodeError, RecursionError, ValueError):
            return []
        if key == "score" and _finite_score(value) is None:
            return []
        i = _skip_whitespace(i)
        if i >= len(text) or text[i] != ",":
            return []
        i += 1

    if array_start is None:
        return []

    results = []
    i = array_start + 1
    while True:
        i = _skip_whitespace(i)
        if i >= len(text):
            return results
        if text[i] != "{":
            return []
        try:
            entry, i = decoder.raw_decode(text, i)
        except (json.JSONDecodeError, RecursionError, ValueError):
            return results if _is_append_only_json_object_prefix(text[i:]) else []
        if not isinstance(entry, dict):
            return []
        results.append(entry)
        i = _skip_whitespace(i)
        if i >= len(text):
            return results
        # Salvage is only for an array truncated before its closing bracket.
        # A closed results array with a malformed outer object is not partial
        # per-entry output and must take the structured-error path.
        if text[i] == "]":
            return []
        if text[i] != ",":
            return []
        i = _skip_whitespace(i + 1)
        if i >= len(text):
            return results
        if text[i] == "]":
            return []


def _resolve_acceptable_skills(entry, expected_skill=None):
    raw = entry.get("acceptable_skills")
    if raw is None:
        raw = entry.get("acceptable_alternates")
    return _accepted_skill_names(expected_skill or entry.get("expected_skill"), raw)


# Keep in sync with skillevaluator.provider_config.MAX_COMPLETION_TOKENS (drift test).
STRUCTURED_JUDGE_MAX_TOKENS = 16384

_JUDGE_RETRY_REMINDER = (
    "\n\nIMPORTANT: Your previous reply could not be parsed or validated. Respond with ONLY the "
    "minified JSON object on a single line -- no markdown fences, no prose, and keep explanations brief."
)


def _call_validated_json_judge(prompt, validate, call, extract, **call_kwargs):
    """Invoke a JSON judge with one format-correction retry when payload validation fails."""
    call_kwargs.setdefault("max_tokens", STRUCTURED_JUDGE_MAX_TOKENS)

    def invoke(call_prompt):
        content, error, *metadata = call(call_prompt, **call_kwargs)
        provenance = metadata[0] if metadata and isinstance(metadata[0], dict) else {}
        parsed = extract(content) if content else None
        validation_error = validate(parsed) if not error else None
        return parsed, error, provenance, validation_error

    parsed, error, provenance, validation_error = invoke(prompt)
    if error:
        return None, f"LLM judge error: {error}", provenance
    if validation_error is None:
        return parsed, None, provenance

    parsed, error, provenance, validation_error = invoke(prompt + _JUDGE_RETRY_REMINDER)
    if error:
        return None, f"LLM judge retry error: {error}", provenance
    if validation_error is not None:
        return None, f"{validation_error} after retry", provenance
    return parsed, None, provenance


# ── LLM Judge: Accuracy (5-criterion) ────────────────────────────────────────


_ACCURACY_CRITERIA_KEYS = frozenset(
    {
        "SKILL_IDENTIFIED",
        "ACTION_CORRECT",
        "FACTUALLY_ACCURATE",
        "TASK_ADDRESSED",
        "ACTIONABLE",
    }
)


def _valid_accuracy_criteria(value):
    """Return True when value is a complete 5-criterion boolean mapping."""
    return (
        isinstance(value, dict)
        and value.keys() == _ACCURACY_CRITERIA_KEYS
        and all(isinstance(item, bool) for item in value.values())
    )


def _accuracy_payload_error(parsed):
    """Validate a parsed accuracy judge payload and return an error message if malformed."""
    if not isinstance(parsed, dict):
        return "Judge response was not a valid JSON object"
    if "reason" in parsed and not isinstance(parsed["reason"], str):
        return "Judge response contained an invalid accuracy reason"
    criteria = parsed.get("criteria")
    criteria_valid = _valid_accuracy_criteria(criteria)
    if "criteria" in parsed and not criteria_valid:
        return "Judge response contained invalid accuracy criteria"
    if _finite_score(parsed.get("score")) is None and not criteria_valid:
        return "Judge response contained no valid accuracy score or complete criteria"
    return None


def _goal_payload_error(parsed):
    """Validate a parsed goal-accuracy judge payload and return an error message if malformed."""
    if not isinstance(parsed, dict):
        return "Judge response was not a valid JSON object"
    for field in ("reason", "user_goal", "end_state"):
        if field in parsed and not isinstance(parsed[field], str):
            return f"Judge response contained an invalid {field} value"
    if not isinstance(parsed.get("achieved"), bool):
        return "Judge response contained an invalid achieved value"
    if "score" in parsed and _finite_score(parsed["score"]) is None:
        return "Judge response contained an invalid goal score"
    return None


ACCURACY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "criteria": {
            "type": "object",
            "properties": {
                "SKILL_IDENTIFIED": {"type": "boolean"},
                "ACTION_CORRECT": {"type": "boolean"},
                "FACTUALLY_ACCURATE": {"type": "boolean"},
                "TASK_ADDRESSED": {"type": "boolean"},
                "ACTIONABLE": {"type": "boolean"},
            },
            "required": [
                "SKILL_IDENTIFIED",
                "ACTION_CORRECT",
                "FACTUALLY_ACCURATE",
                "TASK_ADDRESSED",
                "ACTIONABLE",
            ],
            "additionalProperties": False,
        },
        "score": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["criteria", "score", "reason"],
    "additionalProperties": False,
}

GOAL_ACCURACY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "user_goal": {"type": "string"},
        "end_state": {"type": "string"},
        "achieved": {"type": "boolean"},
        "score": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["user_goal", "end_state", "achieved", "score", "reason"],
    "additionalProperties": False,
}

BEHAVIOR_CHECK_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step": {"type": "integer"},
                    "passed": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["step", "passed", "reason"],
                "additionalProperties": False,
            },
        },
        "score": {"type": "number"},
        "summary": {"type": "string"},
    },
    "required": ["results", "score", "summary"],
    "additionalProperties": False,
}


def judge_accuracy(question, ground_truth, agent_text, *, skill_identification_applicable=True):
    if not _has_judge_reference(ground_truth):
        return _judge_not_applicable(_NO_GROUND_TRUTH_REASON)
    prompt = """You are an expert evaluator for AI agent responses. Evaluate by checking \
each criterion below against the expected answer. For each criterion, determine true (satisfied) or false (not satisfied).

1. SKILL_IDENTIFIED: Does the response reference or use the correct skill for the task?
2. ACTION_CORRECT: Does the response describe or execute the correct actions/scripts?
3. FACTUALLY_ACCURATE: Are the factual claims consistent with the expected answer?
4. TASK_ADDRESSED: Does the response directly address the user's request?
5. ACTIONABLE: Does the response provide actionable information (not just acknowledgment)?

Compute score = count(true) / 5.
Be lenient on exact wording but strict on factual correctness.

Respond with ONLY a JSON object:
{{"criteria": {{"SKILL_IDENTIFIED": true, "ACTION_CORRECT": true, "FACTUALLY_ACCURATE": true, "TASK_ADDRESSED": true, "ACTIONABLE": true}}, "score": 1.0, "reason": "brief summary"}}

USER QUESTION:
{question}

EXPECTED ANSWER:
{ground_truth}

SELECTED EVIDENCE (final response + produced artifacts; low-relevance steps may be omitted):
{agent_text}"""

    if not skill_identification_applicable:
        prompt = prompt.replace(
            "1. SKILL_IDENTIFIED: Does the response reference or use the correct skill for the task?",
            "1. SKILL_IDENTIFIED: Inapplicable: no skill is expected. Set false; exclude it from the score.",
        ).replace("Compute score = count(true) / 5.", "Compute score = count(true among the other four criteria) / 4.")
        prompt = prompt.replace('"SKILL_IDENTIFIED": true', '"SKILL_IDENTIFIED": false')
    # Adjust the rubric before interpolation so task/evidence text stays intact.
    prompt = prompt.format(question=question, ground_truth=ground_truth, agent_text=agent_text)
    prompt = append_character_facts(prompt, question=question, ground_truth=ground_truth, agent_text=agent_text)

    def validate_payload(parsed):
        error = _accuracy_payload_error(parsed)
        if not error and not skill_identification_applicable and not _valid_accuracy_criteria(parsed.get("criteria")):
            return "Accuracy requires complete criterion verdicts when skill identification is inapplicable"
        return error

    parsed, error, _provenance = _call_validated_json_judge(
        prompt,
        validate_payload,
        call_public_llm,
        extract_json,
        response_schema=ACCURACY_JSON_SCHEMA,
        schema_name="accuracy_judgment",
    )
    if error:
        return _judge_error(error)

    assert isinstance(parsed, dict)
    criteria = parsed.get("criteria")
    criteria_valid = _valid_accuracy_criteria(criteria)

    if criteria_valid:
        # Score only applicable verdicts; excluded criteria leave both numerator and denominator.
        excluded = set() if skill_identification_applicable else {"SKILL_IDENTIFIED"}
        applicable = [verdict for name, verdict in criteria.items() if name not in excluded]
        score = sum(verdict is True for verdict in applicable) / len(applicable)
    else:
        # Legacy score-only payloads; validate_payload rejects them when a criterion is excluded.
        assert skill_identification_applicable
        score = _finite_score(parsed.get("score"))
        assert score is not None
    result = {
        "score": round(score, 4),
        "reason": _bounded_judge_text(parsed.get("reason", "")),
        "criteria": criteria if criteria_valid else {},
    }
    if not skill_identification_applicable:
        result["excluded_criteria"] = ["SKILL_IDENTIFIED"]
        result["scoring_policy"] = "applicable-accuracy-criteria/v1"
    return result


# ── LLM Judge: Goal Accuracy ─────────────────────────────────────────────────


def judge_goal_accuracy(question, ground_truth, agent_text, tool_summary=""):
    if not _has_judge_reference(ground_truth):
        return _judge_not_applicable(_NO_GROUND_TRUTH_REASON)

    if _ragas_goal_accuracy_enabled():
        try:
            result = _judge_goal_accuracy_ragas(question, ground_truth, agent_text, tool_summary)
            if not isinstance(result, dict) or _finite_score(result.get("score")) is None:
                raise ValueError("RAGAS returned a non-finite goal accuracy score")
            return result
        except Exception as e:
            logger.info("RAGAS not available (%s), using custom prompt", e)
    return _judge_goal_accuracy_custom(question, ground_truth, agent_text, tool_summary)


def _ragas_goal_accuracy_enabled():
    """RAGAS is an OpenAI-only optimization, never an agent-key fallback.

    Reasoning judges reject the sampling settings RAGAS sends, so they use the prompt judge.
    """
    # The prompt judge can append metadata outside the original evidence.
    if character_facts_enabled():
        return False
    if _public_provider() != "openai" or _is_openai_reasoning_model(_selected_judge_model()):
        return False
    try:
        request_url = _resolve_url("openai")
    except ValueError:
        return False
    return _is_native_openai_chat_url("openai", request_url)


def _judge_goal_accuracy_ragas(question, ground_truth, agent_text, tool_summary):
    """Use RAGAS AgentGoalAccuracyWithReference for high-quality two-step evaluation."""
    import asyncio

    from openai import AsyncOpenAI
    from ragas import SingleTurnSample
    from ragas.llms.base import llm_factory
    from ragas.messages import AIMessage as RagasAI
    from ragas.messages import HumanMessage as RagasHuman
    from ragas.metrics.collections import AgentGoalAccuracyWithReference

    if not _ragas_goal_accuracy_enabled():
        raise RuntimeError("RAGAS goal accuracy requires the selected canonical OpenAI provider")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("No OPENAI_API_KEY")

    request_url = _resolve_url("openai").rstrip("/")
    suffix = "/chat/completions"
    if not request_url.endswith(suffix) or not _is_native_openai_chat_url("openai", request_url):
        raise RuntimeError("RAGAS goal accuracy requires the selected canonical OpenAI provider")
    client = AsyncOpenAI(api_key=api_key, base_url=request_url[: -len(suffix)])
    llm = llm_factory(_selected_judge_model(), client=client)

    metric = AgentGoalAccuracyWithReference(llm=llm)
    user_input = [
        RagasHuman(content=question),
        RagasAI(content=agent_text),
    ]

    sample = SingleTurnSample(user_input=user_input, reference=ground_truth)

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            asyncio.wait_for(
                metric.ascore(sample),
                timeout=_remaining_judge_timeout(_resolve_judge_wall_time_budget()),
            )
        )
    finally:
        loop.close()

    score = float(result.value) if hasattr(result, "value") else float(result)
    if not math.isfinite(score):
        raise ValueError("RAGAS returned a non-finite goal accuracy score")
    return {
        "score": max(0.0, min(1.0, score)),
        "reason": "RAGAS AgentGoalAccuracyWithReference",
        "method": "ragas",
        "provider": "openai",
        "model": _selected_judge_model(),
    }


def _judge_goal_accuracy_custom(question, ground_truth, agent_text, tool_summary):
    """Fallback: two-step custom prompt mirroring RAGAS logic."""
    prompt = f"""You are an evaluation judge. Determine whether an AI agent achieved the expected goal.

Step 1: What was the user's goal?
Step 2: What end state did the agent reach?
Step 3: Compare the end state to the expected outcome.

USER REQUEST:
{question}

EXPECTED OUTCOME:
{ground_truth}

AGENT'S TOOL CALLS:
{tool_summary}

END-STATE EVIDENCE:
{agent_text}

Did the agent achieve the expected goal?
Respond with ONLY a JSON object:
{{"user_goal": "...", "end_state": "...", "achieved": true/false, "score": 1.0, "reason": "..."}}"""
    prompt = append_character_facts(
        prompt, question=question, ground_truth=ground_truth, tool_summary=tool_summary, agent_text=agent_text
    )

    parsed, error, provenance = _call_validated_json_judge(
        prompt,
        _goal_payload_error,
        _call_public_llm_with_provenance,
        extract_json,
        response_schema=GOAL_ACCURACY_JSON_SCHEMA,
        schema_name="goal_accuracy_judgment",
    )
    if error:
        return _judge_error(error, **provenance)

    assert isinstance(parsed, dict)
    achieved = parsed.get("achieved")
    assert isinstance(achieved, bool)

    score = 1.0 if achieved else 0.0
    if "score" in parsed:
        score = _finite_score(parsed["score"])
        assert score is not None
    return {
        "score": score,
        "reason": _bounded_judge_text(parsed.get("reason", "")),
        "user_goal": _bounded_judge_text(parsed.get("user_goal", "")),
        "end_state": _bounded_judge_text(parsed.get("end_state", "")),
        "method": "custom",
        **provenance,
    }


# ── LLM Judge: Behavior Check ────────────────────────────────────────────────

# Reasoning judges (e.g. openai/openai/gpt-5*) spend completion budget on hidden
# reasoning tokens before emitting the per-behavior results array; the old 1024
# cap was observed live to truncate behavior_check output to EMPTY content
# (finish_reason="length", reasoning_tokens=1024).
BEHAVIOR_JUDGE_MAX_TOKENS = STRUCTURED_JUDGE_MAX_TOKENS

_BEHAVIOR_RETRY_REMINDER = (
    "\n\nIMPORTANT: Your previous reply could not be parsed. Respond with ONLY the "
    "minified JSON object on a single line -- no markdown fences, no prose, and "
    'keep every "reason" under 15 words.'
)


def judge_behavior_check(conversation_text, expected_behaviors):
    if not _has_judge_reference(expected_behaviors):
        return _judge_not_applicable(_NO_EXPECTED_BEHAVIOR_REASON, results=[])

    behaviors_text = "\n".join(f"{i + 1}. {b}" for i, b in enumerate(expected_behaviors))
    conversation = _compact_behavior_conversation(conversation_text)

    prompt = f"""You are evaluating whether an AI agent followed expected behaviors during a task. \
Analyze the full conversation and determine if each expected behavior was observed.

CONVERSATION:
{conversation}

EXPECTED BEHAVIORS:
{behaviors_text}

For each behavior, set "passed" to true (observed) or false (not observed) with a brief reason.

Respond with ONLY a JSON object:
{{"results": [{{"step": 1, "passed": true, "reason": "..."}}, ...], "score": 0.67, "summary": "brief summary"}}"""
    prompt = append_character_facts(prompt, conversation=conversation, expected_behaviors=behaviors_text)

    content, error = call_public_llm(
        prompt,
        max_tokens=BEHAVIOR_JUDGE_MAX_TOKENS,
        response_schema=BEHAVIOR_CHECK_JSON_SCHEMA,
        schema_name="behavior_check_judgment",
    )
    if error:
        return _judge_error(f"LLM judge error: {error}", results=[])

    def _parse_judge_object(text):
        return extract_json(text) if text else None

    parsed = _parse_judge_object(content)
    score = _behavior_payload_score(parsed, len(expected_behaviors))
    attempts = [(content or "", parsed)]
    retry_error = None
    if score is None:
        # One retry max, with an explicit machine-readable-output reminder.
        retry_content, retry_error = call_public_llm(
            prompt + _BEHAVIOR_RETRY_REMINDER,
            max_tokens=BEHAVIOR_JUDGE_MAX_TOKENS,
            response_schema=BEHAVIOR_CHECK_JSON_SCHEMA,
            schema_name="behavior_check_judgment",
        )
        if not retry_error:
            parsed = _parse_judge_object(retry_content)
            attempts.append((retry_content or "", parsed))
            score = _behavior_payload_score(parsed, len(expected_behaviors))

    if score is None:
        # Salvage complete entries from a truncated results array (newest first).
        for text, extracted in reversed(attempts):
            if extracted is not None:
                continue
            salvaged = _salvage_behavior_results(text)
            if salvaged:
                candidate = {
                    "results": salvaged,
                    "summary": (
                        f"Salvaged {len(salvaged)}/{len(expected_behaviors)} behavior "
                        "results from truncated judge response"
                    ),
                }
                candidate_score = _behavior_payload_score(
                    candidate,
                    len(expected_behaviors),
                    allow_partial=True,
                )
                if candidate_score is not None:
                    parsed = candidate
                    score = candidate_score
                    break

    if score is None:
        if retry_error:
            return _judge_error(f"LLM judge retry error: {retry_error}", results=[])
        return _judge_error("Judge response was unparseable or invalid after retry", results=[])

    results = parsed["results"]
    return {
        "score": round(score, 4),
        "reason": parsed.get("summary", ""),
        "results": results,
    }


def _behavior_payload_score(parsed, expected_count, *, allow_partial=False):
    if not isinstance(parsed, dict):
        return None
    results = parsed.get("results")
    if not isinstance(results, list):
        return None
    if any(not isinstance(result, dict) or not isinstance(result.get("passed"), bool) for result in results):
        return None
    if allow_partial:
        if not results or len(results) > expected_count:
            return None
    elif len(results) != expected_count:
        return None
    if "score" in parsed and _finite_score(parsed["score"]) is None:
        return None
    denominator = expected_count if allow_partial else len(results)
    if denominator <= 0:
        return None
    return sum(1 for result in results if result["passed"]) / denominator


# ── Main ─────────────────────────────────────────────────────────────────────


def _finite_reward_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        numeric = float(value)
    except (OverflowError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _normalize_required_judge_result(metric, result, *, allow_not_applicable=False):
    if isinstance(result, dict):
        normalized = dict(result)
        status = str(result.get("status", "")).casefold()
        status_is_error = status == "error"
        if status == NOT_APPLICABLE_STATUS and allow_not_applicable:
            # Only the verifier decides N/A (the case has no reference input);
            # the score is always null so it can never be averaged as 0 or 1.
            normalized["score"] = None
            normalized["status"] = NOT_APPLICABLE_STATUS
            normalized["reason"] = _bounded_judge_text(result.get("reason")) or f"{metric} is not applicable"
            return normalized
        score_is_valid = _finite_reward_number(result.get("score")) is not None
        if not status_is_error and status != NOT_APPLICABLE_STATUS and score_is_valid:
            return normalized
        supplied_reason = str(result.get("reason") or "").strip()
        if status == NOT_APPLICABLE_STATUS:
            reason = f"Required {metric} judge reported not_applicable although the eval case defines its reference"
        else:
            reason = supplied_reason if status_is_error else f"Required {metric} judge returned an invalid score"
            if supplied_reason and not status_is_error:
                reason = f"{reason}: {supplied_reason}"
    else:
        normalized = {}
        reason = f"Required {metric} judge returned an invalid result"

    normalized["score"] = None
    normalized["status"] = "error"
    normalized["reason"] = _judge_error(reason)["reason"]
    return normalized


def _call_required_judge(metric, judge, *args, allow_not_applicable=False, **kwargs):
    """Run a required LLM judge under a bounded wall-time deadline and normalize its result.

    The result records the model that produced it (``judge_model_used``), the
    reasoning effort sent, and whether a fallback model answered, so lift can
    leave out cases whose two arms were judged by different models.
    """
    previous_deadline = _ACTIVE_JUDGE_DEADLINE.get()
    own_deadline = time.monotonic() + _resolve_judge_wall_time_budget()
    deadline = min(previous_deadline, own_deadline) if previous_deadline is not None else own_deadline
    token = _ACTIVE_JUDGE_DEADLINE.set(deadline)
    calls = []
    calls_token = _ACTIVE_JUDGE_CALLS.set(calls)
    alarm_armed = False
    previous_alarm_handler = None

    # The standalone Harbor verifier runs judges on its main thread. Its
    # interval timer also interrupts a response body that keeps trickling data
    # inside one socket read, where urllib's idle timeout cannot help.
    try:
        try:
            if hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread():
                active_alarm, repeat_interval = signal.getitimer(signal.ITIMER_REAL)
                if active_alarm == 0 and repeat_interval == 0:
                    previous_alarm_handler = signal.getsignal(signal.SIGALRM)

                    def _raise_judge_timeout(_signum, _frame):
                        raise TimeoutError("LLM judge time budget exhausted")

                    signal.signal(signal.SIGALRM, _raise_judge_timeout)
                    try:
                        signal.setitimer(signal.ITIMER_REAL, max(deadline - time.monotonic(), 1e-6))
                        alarm_armed = True
                    except OSError:
                        signal.signal(signal.SIGALRM, previous_alarm_handler)
            result = judge(*args, **kwargs)
        finally:
            if alarm_armed:
                try:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                finally:
                    signal.signal(signal.SIGALRM, previous_alarm_handler)
    except Exception as exc:
        result = _judge_error(f"Required {metric} judge raised {type(exc).__name__}: {exc}")
    finally:
        _ACTIVE_JUDGE_DEADLINE.reset(token)
        _ACTIVE_JUDGE_CALLS.reset(calls_token)
    normalized = _normalize_required_judge_result(metric, result, allow_not_applicable=allow_not_applicable)
    if calls:
        normalized["judge_model_used"] = calls[-1]["model"]
        normalized["judge_reasoning_effort"] = calls[-1]["reasoning_effort"]
        normalized["judge_fallback_used"] = any(call["fallback_used"] for call in calls)
    return normalized


NOT_APPLICABLE_STATUS = "not_applicable"


_NO_GROUND_TRUTH_REASON = "N/A: no ground_truth defined for this eval case"


_NO_EXPECTED_BEHAVIOR_REASON = "N/A: no expected_behavior defined for this eval case"


_JUDGED_METRICS = {
    "accuracy": ("ground_truth", _NO_GROUND_TRUTH_REASON),
    "goal_accuracy": ("ground_truth", _NO_GROUND_TRUTH_REASON),
    "behavior_check": ("expected_behavior", _NO_EXPECTED_BEHAVIOR_REASON),
}


SKILL_METRICS_NOT_APPLICABLE_REASON = (
    "N/A: this arm has no skill under test installed, so there is no skill to discover, run or route to"
)


_SCORE_POLICY_DIMENSIONS = (
    ("security",),
    ("accuracy",),
    ("skill_execution",),
    ("goal_accuracy", "behavior_check"),
    ("skill_efficiency",),
)


def _judge_not_applicable(reason, **metadata):
    """Return a scoreless result for a judge that has nothing to judge against."""
    return {**metadata, "score": None, "status": NOT_APPLICABLE_STATUS, "reason": reason}


def _has_judge_reference(value):
    """Return whether a ground_truth / expected_behavior value gives a judge something to check."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return any(_has_judge_reference(item) for item in value)
    if isinstance(value, (bool, int, float)):
        # 0, 0.0 and False are real reference answers, not a missing one.
        return True
    return bool(value)


def skill_metrics_applicable(
    has_skill,
    should_trigger,
    expected_skill,
    acceptable_skills=None,
    *,
    evaluated_skill=None,
    workspace_skill_names=None,
):
    """Return whether an arm can be scored on ``skill_execution`` and ``skill_efficiency``.

    The arm that carries the skill or plugin always can. An arm without it (the
    no-skill or no-plugin baseline) can only when it stages a skill the case
    could activate: an acceptable alternate in a skill workspace. Otherwise both metrics are not
    applicable, so skill activation alone cannot earn lift.
    """
    if has_skill is not False:
        return True
    staged = {str(name).strip().lower() for name in workspace_skill_names or [] if str(name).strip()}
    if not staged:
        return False
    if should_trigger is False:
        candidates = [evaluated_skill or expected_skill]
    else:
        candidates = [expected_skill, *(acceptable_skills or [])]
    return any(str(name).strip().lower() in staged for name in candidates if name and str(name).strip())


def _skill_metric_not_applicable():
    """Scoreless ``skill_execution`` / ``skill_efficiency`` result for an arm without the skill."""
    return {"score": None, "status": NOT_APPLICABLE_STATUS, "reason": SKILL_METRICS_NOT_APPLICABLE_REASON}


def _reward_overall(result, details):
    """The score policy's mean of the five dimensions, leaving out judged metrics recorded as N/A.

    An N/A metric is left out of its dimension, and a dimension with no scored
    metric is left out of the mean.
    """

    def scored(metric):
        return not (
            metric != "security"
            and result.get(metric) is None
            and isinstance(details.get(metric), dict)
            and details[metric].get("status") == NOT_APPLICABLE_STATUS
        )

    dimensions = []
    for metrics in _SCORE_POLICY_DIMENSIONS:
        values = [float(result[metric]) for metric in metrics if scored(metric)]
        if values:
            dimensions.append(round(sum(values) / len(values), 4))
    return round(sum(dimensions) / len(dimensions), 4)


def _format_log_score(value):
    numeric = _finite_reward_number(value)
    return f"{numeric:.2f}" if numeric is not None else "N/A"


def _numeric_reward_payload(result, overall):
    payload = {}
    for key, value in result.items():
        numeric = _finite_reward_number(value)
        if numeric is not None:
            payload[key] = numeric
    numeric_overall = _finite_reward_number(overall)
    if numeric_overall is not None:
        payload["overall"] = numeric_overall
    return payload


def write_reward_outputs(result, overall):
    # Top-level result keys are the fixed verifier schema. Sanitize their values
    # recursively so credential text cannot rename Harbor's reward metrics.
    sanitized_result = {key: _sanitize_error_value(value) for key, value in result.items()}
    sanitized_overall = _sanitize_error_value(overall)
    REWARD_JSON.parent.mkdir(parents=True, exist_ok=True)
    skill_evaluator_reward_json = SKILL_EVALUATOR_REWARD_JSON
    if (
        skill_evaluator_reward_json == VERIFIER_DIR / "skill_evaluator_reward.json"
        and REWARD_JSON.parent != VERIFIER_DIR
    ):
        skill_evaluator_reward_json = REWARD_JSON.parent / skill_evaluator_reward_json.name
    skill_evaluator_reward_json.parent.mkdir(parents=True, exist_ok=True)
    skill_evaluator_reward_json.write_text(json.dumps(sanitized_result, indent=2))
    REWARD_JSON.write_text(json.dumps(_numeric_reward_payload(sanitized_result, sanitized_overall), indent=2))
    REWARD_TXT.write_text(str(sanitized_overall))


def main():
    entry = json.loads(ENTRY_PATH.read_text(encoding="utf-8"))
    traj, traj_meta = load_trajectory_with_fallback(ATIF_PATH, ATIF_PATH.parent)

    expected_skill = entry.get("expected_skill") or ""
    expected_script = entry.get("expected_script") or ""
    should_trigger = resolve_should_trigger(entry)
    evaluated_skill = entry.get("evaluated_skill") or ""
    acceptable_skills = _resolve_acceptable_skills(entry, expected_skill)
    expected_behavior = entry.get("expected_behavior", [])
    question = entry.get("question", "")
    ground_truth = entry.get("ground_truth", "")
    workspace_mode = entry.get("skill_workspace_mode", "isolated")
    workspace_skill_names = entry.get("workspace_skill_names", [])
    if not isinstance(workspace_skill_names, list):
        workspace_skill_names = []

    skill_metrics_on = skill_metrics_applicable(
        entry.get("has_skill", True),
        should_trigger,
        expected_skill,
        acceptable_skills,
        evaluated_skill=evaluated_skill,
        workspace_skill_names=workspace_skill_names,
    )

    if not traj:
        # Without a trajectory every metric the case defines fails, but a judge
        # without its reference input stays N/A exactly as on the judged path,
        # so one crashed trial cannot turn an arm's N/A judge metric into 0.0.
        judge_scores = {}
        judge_details = {}
        for metric, (field, na_reason) in _JUDGED_METRICS.items():
            if _has_judge_reference(entry.get(field)):
                judge_scores[metric] = 0
            else:
                judge_scores[metric] = None
                judge_details[metric] = _judge_not_applicable(na_reason)
        skill_scores = {"skill_execution": 0, "skill_efficiency": 0}
        if not skill_metrics_on:
            skill_scores = {"skill_execution": None, "skill_efficiency": None}
            judge_details.update(
                skill_execution=_skill_metric_not_applicable(),
                skill_efficiency=_skill_metric_not_applicable(),
            )
        result = {
            "security": 0,
            **skill_scores,
            **judge_scores,
            "metric_set": DEFAULT_METRIC_SET,
            "score_policy": DEFAULT_SCORE_POLICY,
            "error": "No trajectory or reconstructible agent log",
            "trajectory_source": traj_meta.get("source"),
            "trajectory_detail": traj_meta.get("warning") or traj_meta.get("note"),
        }
        if judge_details:
            result["details"] = judge_details
        write_reward_outputs(result, 0.0)
        return

    tool_calls = extract_tool_calls_as_dicts(traj)
    skill_tools = get_skill_tool_calls(traj)

    details: dict[str, Any] = {}
    if traj_meta.get("note") or traj_meta.get("warning") or traj_meta.get("source") != "trajectory.json":
        details["_trajectory_load"] = {
            "source": traj_meta.get("source"),
            "note": traj_meta.get("note"),
            "warning": traj_meta.get("warning"),
        }
    if len(acceptable_skills) > 1:
        details["_skill_routing_policy"] = {
            "expected_skill": expected_skill,
            "acceptable_skills": acceptable_skills,
            "acceptable_alternates": acceptable_skills[1:],
            "alternate_score": ACCEPTABLE_ALTERNATE_SCORE,
        }

    # ── Eval 1: security ─────────────────────────────────────────────────
    security_result = check_security(traj, tool_calls, expected_skill, acceptable_skills)
    security_score = security_result["score"]
    details["security"] = security_result

    # ── Eval 2: skill_execution ──────────────────────────────────────────
    if skill_metrics_on:
        skill_execution_result = score_skill_execution(
            tool_calls,
            expected_skill,
            expected_script,
            should_trigger,
            evaluated_skill=evaluated_skill,
            require_evaluated_skill=True,
            skill_tool_names=skill_tools,
            acceptable_skills=acceptable_skills,
        )
        se_score = skill_execution_result["score"]
        details["skill_execution"] = skill_execution_result["details"]
    else:
        se_score = None
        details["skill_execution"] = _skill_metric_not_applicable()

    # ── Eval 3: skill_efficiency ─────────────────────────────────────────
    if not skill_metrics_on:
        sef_score = None
        details["skill_efficiency"] = _skill_metric_not_applicable()
    elif not should_trigger or not expected_skill:
        sef_score = 1.0
        details["skill_efficiency"] = {"message": "Skipped (negative or no expected_skill)"}
    elif not tool_calls:
        sef_score = 0.0
        details["skill_efficiency"] = {"message": "No tool calls in trajectory"}
    else:
        checks = {}
        scores = []
        r = check_routing(
            tool_calls,
            expected_skill,
            skill_tool_names=skill_tools,
            workspace_skill_names=workspace_skill_names,
            workspace_mode=workspace_mode,
            acceptable_skills=acceptable_skills,
        )
        checks["routing"] = r
        scores.append(r["score"])
        r = check_tool_efficiency(tool_calls, expected_skill, expected_script)
        checks["tool_efficiency"] = r
        scores.append(r["score"])
        sef_score = round(sum(scores) / len(scores), 4)
        details["skill_efficiency"] = checks

    bundles = build_metric_evidence_bundles(
        traj, question, ground_truth=ground_truth, expected_behavior=expected_behavior
    )

    # ── Eval 4: accuracy (LLM judge) ─────────────────────────────────────
    acc_result = _call_required_judge(
        "accuracy",
        judge_accuracy,
        question,
        ground_truth,
        bundles["accuracy"]["prompt_evidence"],
        skill_identification_applicable=should_trigger and bool(expected_skill),
        allow_not_applicable=not _has_judge_reference(ground_truth),
    )
    acc_score = acc_result["score"]
    details["accuracy"] = acc_result

    # ── Eval 5: goal_accuracy (RAGAS or custom LLM judge) ────────────────
    ga_result = _call_required_judge(
        "goal_accuracy",
        judge_goal_accuracy,
        question,
        ground_truth,
        bundles["goal_accuracy"]["prompt_evidence"],
        tool_summary="",
        allow_not_applicable=not _has_judge_reference(ground_truth),
    )
    ga_score = ga_result["score"]
    details["goal_accuracy"] = ga_result

    # ── Eval 6: behavior_check (LLM judge) ───────────────────────────────
    bc_result = _call_required_judge(
        "behavior_check",
        judge_behavior_check,
        bundles["behavior_check"]["prompt_evidence"],
        expected_behavior,
        allow_not_applicable=not _has_judge_reference(expected_behavior),
    )
    bc_score = bc_result["score"]
    details["behavior_check"] = bc_result

    # persist refs + omission metadata onto the metric details
    attach_metric_evidence_refs(details, {m: bundles[m]["evidence_refs"] for m in bundles})
    for _m, _b in bundles.items():
        if isinstance(details.get(_m), dict):
            details[_m]["omitted"] = _b["omitted"]

    # ── Write results ────────────────────────────────────────────────────
    result: dict[str, Any] = {
        "security": security_score,
        "skill_execution": se_score,
        "skill_efficiency": sef_score,
        "accuracy": acc_score,
        "goal_accuracy": ga_score,
        "behavior_check": bc_score,
        "metric_set": DEFAULT_METRIC_SET,
        "score_policy": DEFAULT_SCORE_POLICY,
        "entry_id": entry.get("id"),
        "has_skill": entry.get("has_skill", True),
        "trajectory_source": traj_meta.get("source"),
        "details": details,
    }

    judge_errors = {
        metric: details[metric]["reason"]
        for metric in ("accuracy", "goal_accuracy", "behavior_check")
        if details[metric].get("status") == "error"
    }
    if judge_errors:
        result["evaluation_status"] = "failed"
        result["evaluation_errors"] = judge_errors
        # Harbor 0.22 still parses reward.json when the verifier exits nonzero.
        # Keep this artifact deliberately incomplete so the collector cannot
        # score it even if the richer diagnostic sidecar is unavailable.
        write_reward_outputs(result, 0.0)
        logger.error("Required LLM judging failed for: %s", ", ".join(sorted(judge_errors)))
        raise SystemExit(1)

    overall = _reward_overall(result, details)

    write_reward_outputs(result, overall)

    logger.info(
        "Scores: security=%s skill_exec=%s efficiency=%s accuracy=%s goal=%s behavior=%s overall=%s",
        *(_format_log_score(result[metric]) for metric in DISPLAY_METRICS),
        _format_log_score(overall),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
