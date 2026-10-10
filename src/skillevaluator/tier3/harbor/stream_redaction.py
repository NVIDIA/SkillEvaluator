# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boundary-safe exact-value redaction for streamed Harbor output."""

from __future__ import annotations

import codecs
import re
import unicodedata
from collections import Counter, deque
from collections.abc import Iterable
from operator import itemgetter

_REDACTION_LABEL = "<redacted>"
_REDACTION_SENTINEL_CANDIDATES = ("␟", "␞", "␝", "␜", "")
_MAX_REDACTION_SCAN_CHUNK = 64 * 1024
MAX_COMMAND_OUTPUT_BYTES = 16 * 1024 * 1024

_OVERFLOW_ANALYSIS_WINDOW_BYTES = 1024 * 1024
_OVERFLOW_EXCERPT_CHARS = 512
_OVERFLOW_LINE_CHARS = 160
_OVERFLOW_TOP_LINES = 3
_OVERFLOW_MAX_TRACKED_LINES = 1024
_OVERFLOW_WINDOW_LINE_CANDIDATES = 64
# Excerpts and lines keep this much extra context so the generic, regex-based
# credential patterns see a whole match before the display cut.
_OVERFLOW_GENERIC_CONTEXT_CHARS = 4096
_OVERFLOW_LINE_CONTEXT_CHARS = 1024
_overflow_line_key = itemgetter(slice(None, _OVERFLOW_LINE_CONTEXT_CHARS))
_LINE_SEPARATOR_RE = re.compile(r"\r\n|\r|\n")
# ANSI CSI/OSC/two-character escapes, then any other C0/C1 control except the
# tab and line separators that the overflow analysis itself interprets.
_TERMINAL_CONTROL_RE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)?|[@-Z\\-_])"
    r"|\x9b[0-?]*[ -/]*[@-~]"
    r"|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)


class CommandOutputLimitError(RuntimeError):
    """Signal that an untrusted command exceeded its raw output budget."""


class CommandOutputByteBudget:
    """Enforce one combined raw stdout/stderr budget for a command."""

    def __init__(self, limit_bytes: int | None = None) -> None:
        self.limit_bytes = MAX_COMMAND_OUTPUT_BYTES if limit_bytes is None else limit_bytes
        if self.limit_bytes <= 0:
            raise ValueError("command output byte limit must be positive")
        self.consumed_bytes = 0

    def consume(self, chunk: bytes) -> None:
        next_total = self.consumed_bytes + len(chunk)
        if next_total > self.limit_bytes:
            raise CommandOutputLimitError(f"Command output exceeded the {self.limit_bytes}-byte safety limit")
        self.consumed_bytes = next_total


class _OverflowLineCensus:
    """Count CR/LF-delimited lines of redacted text with bounded memory.

    Each feed is counted with C-level ``Counter`` work; only lines already
    tracked and that feed's most frequent candidates are merged, and the
    census is pruned back to a fixed number of distinct (truncated) lines.
    """

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self._pending = ""

    def feed(self, text: str) -> None:
        combined = self._pending + text
        held_separator = ""
        if combined.endswith("\r"):
            # A CRLF may straddle two feeds; keep the CR for the next one.
            combined, held_separator = combined[:-1], "\r"
        lines = _LINE_SEPARATOR_RE.split(combined)
        self._pending = _overflow_line_key(lines.pop()) + held_separator
        if lines:
            self._merge(Counter(map(_overflow_line_key, lines)))

    def finish(self) -> None:
        if self._pending:
            self._merge(Counter((self._pending.removesuffix("\r"),)))
        self._pending = ""

    def _merge(self, window_counts: Counter[str]) -> None:
        for line in self.counts.keys() & window_counts.keys():
            self.counts[line] += window_counts[line]
        for line, count in window_counts.most_common(_OVERFLOW_WINDOW_LINE_CANDIDATES):
            if line not in self.counts:
                self.counts[line] = count
        if len(self.counts) > _OVERFLOW_MAX_TRACKED_LINES:
            self.counts = Counter(dict(self.counts.most_common(_OVERFLOW_MAX_TRACKED_LINES)))


def _overflow_display(text: str, limit: int) -> str:
    if len(text) > limit:
        text = text[:limit] + "…"
    return repr(text)


def describe_command_output_overflow(
    accepted_output: bytes | bytearray,
    *,
    rejected_chunk_bytes: int,
    secret_values: Iterable[str] = (),
) -> str:
    """Summarize output accepted before a budget overflow on one bounded line.

    Byte, newline and carriage-return counts describe the raw output.  Every
    excerpt comes from one contiguous pass that first strips terminal control
    sequences and then applies the streaming exact-value and known-shape
    redaction used for live command output, so an excerpt boundary cannot
    split such a secret and control-character removal cannot reassemble one.
    The generic credential patterns applied to logs run on a bounded context
    around each excerpt and line before it is cut, and once more on the
    rendered summary.  Memory stays bounded by the analysis window, the
    excerpt contexts and a capped census of distinct lines.
    """
    from skillevaluator.utils.redaction import redact_sensitive_text

    secrets = tuple(dict.fromkeys(value for value in secret_values if value))
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    redactor = StreamingLogRedactor(secrets)
    census = _OverflowLineCensus()
    excerpt_context = _OVERFLOW_EXCERPT_CHARS + _OVERFLOW_GENERIC_CONTEXT_CHARS
    head = ""
    tail = ""
    redacted_chars = 0
    control_sequences = 0
    replacement_characters = 0

    def sanitize(text: str) -> str:
        nonlocal control_sequences, replacement_characters
        replacement_characters += text.count("\ufffd")
        sanitized, removed = _TERMINAL_CONTROL_RE.subn("", text)
        control_sequences += removed
        return sanitized.replace("\t", " ")

    def analyze(redacted: str) -> None:
        nonlocal head, tail, redacted_chars
        if not redacted:
            return
        if len(head) < excerpt_context:
            head += redacted[: excerpt_context - len(head)]
        tail = (tail + redacted[-excerpt_context:])[-excerpt_context:]
        redacted_chars += len(redacted)
        census.feed(redacted)

    with memoryview(accepted_output) as data:
        for start in range(0, len(data), _OVERFLOW_ANALYSIS_WINDOW_BYTES):
            with data[start : start + _OVERFLOW_ANALYSIS_WINDOW_BYTES] as window:
                analyze(redactor.feed(sanitize(decoder.decode(window))))
    analyze(redactor.feed(sanitize(decoder.decode(b"", final=True))))
    analyze(redactor.finish())
    census.finish()

    newline_count = accepted_output.count(b"\n")
    carriage_return_count = accepted_output.count(b"\r")
    crlf_count = accepted_output.count(b"\r\n")
    top_lines = ", ".join(
        f"{count}x {_overflow_display(redact_sensitive_text(line), _OVERFLOW_LINE_CHARS)}"
        for line, count in census.counts.most_common(_OVERFLOW_TOP_LINES)
    )
    parts = [
        f"accepted {len(accepted_output)} bytes before a {rejected_chunk_bytes}-byte read",
        f"{newline_count} newline-terminated lines",
        f"{carriage_return_count} carriage returns ({crlf_count} in CRLF)",
        f"{control_sequences} terminal control sequences stripped",
        f"{replacement_characters} UTF-8 decode replacements",
        f"top lines: {top_lines or 'none'}",
    ]
    head = redact_sensitive_text(head)
    # Short output fits entirely in the head context.
    whole_output = redacted_chars <= excerpt_context
    tail = head if whole_output else redact_sensitive_text(tail)
    if whole_output and len(head) <= 2 * _OVERFLOW_EXCERPT_CHARS:
        parts.append(f"output: {_overflow_display(head, 2 * _OVERFLOW_EXCERPT_CHARS)}")
    else:
        parts.append(f"head: {_overflow_display(head, _OVERFLOW_EXCERPT_CHARS)}")
        parts.append(f"tail: {_overflow_display(tail[-_OVERFLOW_EXCERPT_CHARS:], _OVERFLOW_EXCERPT_CHARS)}")
    summary = "output diagnostics (redacted): " + "; ".join(parts)
    final_redactor = StreamingLogRedactor(secrets)
    return redact_sensitive_text(final_redactor.feed(summary) + final_redactor.finish())


def diagnosed_command_output_limit_error(
    error: CommandOutputLimitError,
    accepted_output: bytes | bytearray,
    *,
    rejected_chunk_bytes: int,
    secret_values: Iterable[str] = (),
) -> CommandOutputLimitError:
    """Return *error* extended with redacted overflow diagnostics.

    The message still starts with the original limit message.  A diagnostic
    failure only degrades the summary; it never masks the limit failure.
    """
    try:
        diagnostics = describe_command_output_overflow(
            accepted_output,
            rejected_chunk_bytes=rejected_chunk_bytes,
            secret_values=secret_values,
        )
    except Exception as diagnostic_error:
        diagnostics = f"output diagnostics unavailable ({type(diagnostic_error).__name__})"
    return CommandOutputLimitError(f"{error}; {diagnostics}")


def collision_safe_redaction_marker(secret_values: Iterable[str]) -> str:
    """Build a readable marker that cannot contain or join into a secret."""
    secrets = sorted({value for value in secret_values if value})
    if not secrets:
        return _REDACTION_LABEL

    sentinel = next(
        (
            candidate
            for candidate in _REDACTION_SENTINEL_CANDIDATES
            if all(candidate not in secret for secret in secrets)
        ),
        None,
    )
    if sentinel is None:
        used_characters = set().union(*map(set, secrets))
        private_use_ranges = (
            range(0xE000, 0xF900),
            range(0xF0000, 0xFFFFE),
            range(0x100000, 0x10FFFE),
        )
        for candidate_range in private_use_ranges:
            sentinel = next(
                (chr(codepoint) for codepoint in candidate_range if chr(codepoint) not in used_characters),
                None,
            )
            if sentinel is not None:
                break
        if sentinel is None:
            scalar_ranges = (range(1, 0xD800), range(0xE000, 0x110000))
            for scalar_range in scalar_ranges:
                sentinel = next(
                    (
                        chr(codepoint)
                        for codepoint in scalar_range
                        if chr(codepoint) not in used_characters and unicodedata.category(chr(codepoint))[0] in "LNPS"
                    ),
                    None,
                )
                if sentinel is not None:
                    break
    if sentinel is None:
        raise RuntimeError("Could not construct a collision-safe redaction marker")

    minimum_secret_length = min(map(len, secrets))
    if minimum_secret_length == 1:
        return sentinel
    chunk_length = minimum_secret_length - 1
    label_chunks = [
        _REDACTION_LABEL[index : index + chunk_length] for index in range(0, len(_REDACTION_LABEL), chunk_length)
    ]
    return sentinel + sentinel.join(label_chunks) + sentinel


class _SecretTrieNode:
    __slots__ = ("children", "depth", "failure", "max_terminal_length", "terminal_length")

    def __init__(self) -> None:
        self.children: dict[str, _SecretTrieNode] = {}
        self.depth = 0
        self.failure = self
        self.max_terminal_length = 0
        self.terminal_length = 0


class StreamingSecretRedactor:
    """Redact exact-value match unions with partition-linear work.

    This is an Aho-Corasick matcher: each character advances through at most
    one successful edge plus amortized failure edges, independent of secret
    count and input partitioning.  When the automaton is at its root, a C-level
    character-class search skips spans that cannot begin any secret.  Coverage
    and output are committed in slices, keeping dense short-secret output both
    memory-bounded and fast.
    """

    def __init__(self, secret_values: Iterable[str]) -> None:
        secrets = sorted({value for value in secret_values if value})
        self._root = _SecretTrieNode()
        for secret in secrets:
            node = self._root
            for character in secret:
                child = node.children.get(character)
                if child is None:
                    child = _SecretTrieNode()
                    child.depth = node.depth + 1
                    node.children[character] = child
                node = child
            node.terminal_length = len(secret)

        failure_queue = deque(self._root.children.values())
        for child in failure_queue:
            child.failure = self._root
            child.max_terminal_length = child.terminal_length
        while failure_queue:
            node = failure_queue.popleft()
            for character, child in node.children.items():
                failure = node.failure
                while failure is not self._root and character not in failure.children:
                    failure = failure.failure
                child.failure = failure.children.get(character, self._root)
                child.max_terminal_length = max(child.terminal_length, child.failure.max_terminal_length)
                failure_queue.append(child)

        self._has_secrets = bool(secrets)
        self._max_secret_length = max(map(len, secrets), default=0)
        self._single_character_re = (
            re.compile("[" + "".join(re.escape(secret) for secret in secrets) + "]+")
            if self._max_secret_length == 1
            else None
        )
        self._state = self._root
        self._first_character_re = (
            re.compile("[" + "".join(re.escape(character) for character in self._root.children) + "]")
            if self._root.children
            else None
        )
        self._pending = ""
        self._pending_start = 0
        self._processed = 0
        self._coverage: deque[tuple[int, int]] = deque()
        self._redaction_open = False
        self._replacement = collision_safe_redaction_marker(secrets)

    def _add_coverage(self, start: int, end: int) -> None:
        merged_start = start
        while self._coverage and self._coverage[-1][1] >= merged_start:
            previous_start, _previous_end = self._coverage.pop()
            merged_start = min(merged_start, previous_start)
        self._coverage.append((merged_start, end))

    def _advance(self, character: str) -> None:
        while True:
            child = self._state.children.get(character)
            if child is not None:
                self._state = child
                break
            if self._state is self._root:
                break
            self._state = self._state.failure
        self._processed += 1
        if match_length := self._state.max_terminal_length:
            self._add_coverage(self._processed - match_length, self._processed)

    def _drain(self, *, final: bool) -> str:
        # The automaton state is the longest current suffix that can still
        # grow into a secret.  Everything before that suffix is irrevocably
        # safe; retaining a maximum-secret window would delay callbacks even
        # after a mismatch returned to the root.
        safe_end = self._processed if final else self._processed - self._state.depth
        if safe_end <= self._pending_start:
            return ""

        emitted: list[str] = []
        cursor = self._pending_start
        while self._coverage and self._coverage[0][0] < safe_end:
            start, end = self._coverage[0]
            if start > cursor:
                emitted.append(self._pending[cursor - self._pending_start : start - self._pending_start])
                self._redaction_open = False
            if not self._redaction_open:
                emitted.append(self._replacement)
            self._redaction_open = True
            cursor = min(end, safe_end)
            if end <= safe_end:
                self._coverage.popleft()
            else:
                break
        if cursor < safe_end:
            emitted.append(self._pending[cursor - self._pending_start : safe_end - self._pending_start])
            self._redaction_open = False
        committed = safe_end - self._pending_start
        self._pending = self._pending[committed:]
        self._pending_start = safe_end
        return "".join(emitted)

    def _feed_chunk(self, text: str) -> str:
        offset = 0
        pending_parts = [self._pending]
        while offset < len(text):
            if self._state is self._root:
                first_character_re = self._first_character_re
                if first_character_re is None:
                    raise RuntimeError("exact redactor invariant violated")
                match = first_character_re.search(text, offset)
                end = len(text) if match is None else match.start()
                if end > offset:
                    plain = text[offset:end]
                    pending_parts.append(plain)
                    self._processed += len(plain)
                    offset = end
                    if match is None:
                        break
            character = text[offset]
            pending_parts.append(character)
            self._advance(character)
            offset += 1
        self._pending = "".join(pending_parts)
        return self._drain(final=False)

    def feed(self, text: str, *, final: bool = False) -> str:
        """Return safe text while retaining one maximum-pattern window."""
        if not self._has_secrets:
            return text
        if self._single_character_re is not None:
            emitted: list[str] = []
            cursor = 0
            for match in self._single_character_re.finditer(text):
                if match.start() > cursor:
                    emitted.append(text[cursor : match.start()])
                    self._redaction_open = False
                if not self._redaction_open:
                    emitted.append(self._replacement)
                self._redaction_open = True
                cursor = match.end()
            if cursor < len(text):
                emitted.append(text[cursor:])
                self._redaction_open = False
            return "".join(emitted)

        emitted = "".join(
            self._feed_chunk(text[offset : offset + _MAX_REDACTION_SCAN_CHUNK])
            for offset in range(0, len(text), _MAX_REDACTION_SCAN_CHUNK)
        )
        if final:
            emitted += self._drain(final=True)
        return emitted

    def finish(self) -> str:
        """Flush the suffix once later input cannot complete a secret."""
        return self.feed("", final=True)


class StreamingLogRedactor:
    """Redact known key shapes and exact values across arbitrary chunks.

    Each family runs once on either side of the other.  The post-known pass
    catches a token boundary created by an exact-value marker; the final exact
    pass catches a configured value synthesized by a known-shape replacement.
    Collision-safe exact markers cannot themselves create either family.
    """

    def __init__(self, secret_values: Iterable[str]) -> None:
        self._known_redactor = _StreamingKnownPatternRedactor()
        self._exact_redactor = StreamingSecretRedactor(secret_values)
        self._post_known_redactor = _StreamingKnownPatternRedactor()
        self._post_exact_redactor = StreamingSecretRedactor(secret_values)

    def feed(self, text: str) -> str:
        known = self._known_redactor.feed(text)
        exact = self._exact_redactor.feed(known)
        post_known = self._post_known_redactor.feed(exact)
        return self._post_exact_redactor.feed(post_known)

    def finish(self) -> str:
        known = self._known_redactor.finish()
        exact = self._exact_redactor.feed(known) + self._exact_redactor.finish()
        post_known = self._post_known_redactor.feed(exact) + self._post_known_redactor.finish()
        return self._post_exact_redactor.feed(post_known) + self._post_exact_redactor.finish()


_KNOWN_PREFIX_RE = re.compile(r"sk-|nvapi-|crsr_|sha256~|eyJ")
_KNOWN_PREFIXES = ("sk-", "nvapi-", "crsr_", "sha256~", "eyJ")
_MAX_KNOWN_PREFIX_LENGTH = len("sha256~")
_ASCII_KEY_BODY_RE = re.compile(r"[^A-Za-z0-9_-]")
_ASCII_ALNUM_RE = re.compile(r"[^A-Za-z0-9]")
_ASCII_HEX_RE = re.compile(r"[^a-f0-9]")
_OPENSHIFT_BODY_RE = re.compile(r"[^A-Za-z0-9._~-]")
_JWT_SEGMENT_RE = re.compile(r"[^A-Za-z0-9_-]")
_JWT_TOKEN_RE = re.compile(r"[^A-Za-z0-9_.-]")
_GLUED_KEY_BUFFER_LIMIT = 256
_JWT_CANDIDATE_BUFFER_LIMIT = 256
_KNOWN_REPLACEMENTS = {
    "sk-": "sk-<redacted>",
    "nvapi-": "nvapi-<redacted>",
    "crsr_": "crsr_<redacted>",
    "sha256~": "sha256~<redacted>",
    "eyJ": "jwt-<redacted>",
}


def _is_ascii_alnum(character: str) -> bool:
    return character.isascii() and character.isalnum()


def _is_ascii_key_body(character: str) -> bool:
    return _is_ascii_alnum(character) or character in "_-"


def _is_openshift_body(character: str) -> bool:
    return _is_ascii_alnum(character) or character in "._~-"


def _partial_known_prefix_length(text: str) -> int:
    maximum = min(_MAX_KNOWN_PREFIX_LENGTH - 1, len(text))
    return next(
        (
            length
            for length in range(maximum, 0, -1)
            if any(prefix.startswith(text[-length:]) for prefix in _KNOWN_PREFIXES)
        ),
        0,
    )


class _StreamingKnownPatternRedactor:
    """Recognize known secret shapes without buffering ordinary output.

    Plain chunks are scanned by the regular-expression engine and retain only
    the longest possible partial prefix.  Once a prefix is found, the small
    deterministic recognizer below either rejects it or proves it secret.  A
    proven token emits its marker immediately and discards the remainder with
    a compiled character-class search, so attacker-sized tokens do not grow a
    Python list or monopolize the event loop.

    The glued ``sk-``/``nvapi-`` alternative can otherwise remain ambiguous
    forever while waiting for its lower/upper/digit mix.  After a bounded
    prefix it is conservatively redacted.  That favors non-disclosure for an
    adversarial token while preserving the canonical behavior for normal
    candidates and all boundary-prefixed keys.
    """

    def __init__(self) -> None:
        self._plain_tail = ""
        self._candidate_kind: str | None = None
        self._candidate_prefix = ""
        self._candidate: list[str] = []
        self._candidate_count = 0
        self._candidate_has_lower = False
        self._candidate_has_upper = False
        self._candidate_has_digit = False
        self._jwt_stage = 0
        self._discard_re: re.Pattern[str] | None = None
        self._previous_raw_character = ""

    def _reset_candidate(self) -> None:
        self._candidate_kind = None
        self._candidate_prefix = ""
        self._candidate = []
        self._candidate_count = 0
        self._candidate_has_lower = False
        self._candidate_has_upper = False
        self._candidate_has_digit = False
        self._jwt_stage = 0

    def _start_candidate(self, prefix: str) -> None:
        self._candidate_prefix = prefix
        self._candidate = [prefix]
        boundary_character = self._previous_raw_character
        if prefix in {"sk-", "nvapi-"}:
            self._candidate_kind = (
                "boundary-key" if not boundary_character or not _is_ascii_key_body(boundary_character) else "glued-key"
            )
        elif prefix == "crsr_":
            self._candidate_kind = (
                "crsr" if not boundary_character or not _is_ascii_key_body(boundary_character) else "invalid"
            )
        elif prefix == "sha256~":
            self._candidate_kind = (
                "openshift" if not boundary_character or not _is_ascii_key_body(boundary_character) else "invalid"
            )
        else:
            self._candidate_kind = (
                "jwt"
                if not boundary_character or not (boundary_character.isalnum() or boundary_character == "_")
                else "invalid"
            )

    def _prove_candidate(self, emitted: list[str], discard_re: re.Pattern[str]) -> None:
        replacement = _KNOWN_REPLACEMENTS[self._candidate_prefix]
        emitted.append(replacement)
        # Later canonical redactors see the replacement, not the raw token.
        # Retaining that boundary also catches an adjacent lower-priority shape
        # such as ``crsr_<hex>sha256~...``.
        self._previous_raw_character = replacement[-1]
        self._discard_re = discard_re
        self._reset_candidate()

    def _reject_candidate(self, pending: str, offset: int, emitted: list[str]) -> str:
        raw_candidate = "".join(self._candidate)
        # Emitting one character guarantees progress while allowing every
        # nested prefix in the bounded remainder to be recognized normally.
        emitted.append(raw_candidate[0])
        self._previous_raw_character = raw_candidate[0]
        self._reset_candidate()
        return raw_candidate[1:] + pending[offset:]

    def _consume_candidate(self, pending: str, emitted: list[str], *, final: bool) -> str:
        offset = 0
        while offset < len(pending):
            character = pending[offset]
            kind = self._candidate_kind
            if kind == "invalid":
                return self._reject_candidate(pending, offset, emitted)

            if kind == "boundary-key":
                if not _is_ascii_key_body(character):
                    return self._reject_candidate(pending, offset, emitted)
                self._candidate.append(character)
                self._candidate_count += 1
                offset += 1
                if self._candidate_count == 8:
                    self._prove_candidate(emitted, _ASCII_KEY_BODY_RE)
                    return pending[offset:]
                continue

            if kind == "glued-key":
                if not _is_ascii_alnum(character):
                    return self._reject_candidate(pending, offset, emitted)
                self._candidate.append(character)
                self._candidate_count += 1
                self._candidate_has_lower |= character.islower()
                self._candidate_has_upper |= character.isupper()
                self._candidate_has_digit |= character.isdigit()
                offset += 1
                if self._candidate_count >= 20 and (
                    self._candidate_has_lower and self._candidate_has_upper and self._candidate_has_digit
                ):
                    self._prove_candidate(emitted, _ASCII_ALNUM_RE)
                    return pending[offset:]
                if self._candidate_count == _GLUED_KEY_BUFFER_LIMIT:
                    self._prove_candidate(emitted, _ASCII_ALNUM_RE)
                    return pending[offset:]
                continue

            if kind == "crsr":
                if character not in "abcdef0123456789":
                    return self._reject_candidate(pending, offset, emitted)
                self._candidate.append(character)
                self._candidate_count += 1
                offset += 1
                if self._candidate_count == 16:
                    self._prove_candidate(emitted, _ASCII_HEX_RE)
                    return pending[offset:]
                continue

            if kind == "openshift":
                if not _is_openshift_body(character):
                    return self._reject_candidate(pending, offset, emitted)
                self._candidate.append(character)
                offset += 1
                self._prove_candidate(emitted, _OPENSHIFT_BODY_RE)
                return pending[offset:]

            if kind != "jwt":
                raise RuntimeError("known-pattern redactor invariant violated")
            if _is_ascii_key_body(character):
                self._candidate.append(character)
                self._candidate_count += 1
                offset += 1
                if self._jwt_stage == 2 and self._candidate_count == 20:
                    # Waiting for a Unicode word/non-word boundary would make
                    # the final segment unbounded.  At this point the complete
                    # JWT shape is present, so redact conservatively and stream.
                    self._prove_candidate(emitted, _JWT_SEGMENT_RE)
                    return pending[offset:]
                if self._jwt_stage < 2 and self._candidate_count == _JWT_CANDIDATE_BUFFER_LIMIT:
                    # A stage-zero/stage-one JWT candidate can otherwise stay
                    # ambiguous until attacker-controlled EOF. Conservatively
                    # redact after a bounded prefix, then discard the rest of
                    # token-like continuation with the compiled scanner.
                    self._prove_candidate(emitted, _JWT_TOKEN_RE)
                    return pending[offset:]
                continue
            if character != "." or self._jwt_stage >= 2 or self._candidate_count < 20:
                return self._reject_candidate(pending, offset, emitted)
            self._candidate.append(character)
            self._jwt_stage += 1
            self._candidate_count = 0
            offset += 1

        if final:
            return self._reject_candidate(pending, offset, emitted)
        return ""

    def feed(self, text: str) -> str:
        return self._feed(text, final=False)

    def _feed(self, text: str, *, final: bool) -> str:
        emitted: list[str] = []
        pending = self._plain_tail + text
        self._plain_tail = ""
        while pending or (final and self._candidate_kind is not None):
            if self._discard_re is not None:
                end = self._discard_re.search(pending)
                if end is None:
                    pending = ""
                    continue
                pending = pending[end.start() :]
                self._discard_re = None
                continue

            if self._candidate_kind is not None:
                pending = self._consume_candidate(pending, emitted, final=final)
                continue

            match = _KNOWN_PREFIX_RE.search(pending)
            if match is not None:
                plain = pending[: match.start()]
                if plain:
                    emitted.append(plain)
                    self._previous_raw_character = plain[-1]
                self._start_candidate(match.group())
                pending = pending[match.end() :]
                continue

            retained = 0 if final else _partial_known_prefix_length(pending)
            if retained:
                plain = pending[:-retained]
                self._plain_tail = pending[-retained:]
            else:
                plain = pending
            if plain:
                emitted.append(plain)
                self._previous_raw_character = plain[-1]
            pending = ""
        return "".join(emitted)

    def finish(self) -> str:
        return self._feed("", final=True)
