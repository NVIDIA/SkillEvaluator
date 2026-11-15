# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spot advice that takes a case out of the evaluation dataset.

Tier 3 advice may only fix the skill, never the test: "delete the case for positive-001" or "replace the case with
an easier one" would make a failing case pass by removing the requirement. The check is a heuristic over free text,
so it errs on the side of rejecting: a rejected action falls back to a deterministic remedy.

Detection is linear in the length of the text. Each clause is split into tokens once, and every later check looks at
a bounded window of tokens or at a table lookup, so no input can make it backtrack. A clause is a removal when

* a removal verb ("delete", "exclude", "get rid of", "replace ... with an easier one", ...) governs a case phrase
  with at most a few filler words in between, or a case phrase is the subject of a passive removal
  ("the case should be excluded");
* the phrase names a dataset case: a test/eval/dataset-qualified noun, a known case id next to "case", an
  identifier-shaped name next to "case", or a bare "case"/"entry" that the clause ties to the dataset;
* the phrase is not an implementation branch ("the special case in parser.py"), a failure to repair
  ("the test case failure"), or negated advice ("do not delete the case").
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator

# A line break inside a sentence ("delete the case\nfor positive-001") does not end the clause.
_CLAUSE_BREAK = re.compile(r"[.!?;]\s")
_TOKEN = re.compile(r"\w[\w./'\u2019-]*|[,:;!?()\[\]]")
_WORD_TOKEN = re.compile(r"\w[\w./'\u2019-]*")
_PHRASE_END_TOKENS = frozenset({".", ",", ";", ":", "!", "?", ")"})

# Advice that keeps the case: "Do not delete the case", "fix the step instead of deleting the case". A negator only
# counts when it governs the verb, so the words between the two are limited to modifiers of the verb ("without first
# deleting", "without ever having to delete") and "run without failures by deleting the case" stays a removal.
# Only the last few words in front of the verb are read, so the check does not depend on the length of the text.
_NEGATION_WINDOW = 120
_NEGATION_WORD = re.compile(r"[\w'\u2019]+|[^\w\s]")
_PLAIN_WORD = re.compile(r"\w+")
# "do not"-style negators also cover up to two plain words before the verb: "do not try deleting the case".
_LOOSE_NEGATORS = (("do", "not"), ("don't",), ("don\u2019t",), ("never",), ("avoid",))
# "not only deleting the case but also fixing the parser" recommends the deletion. After one of these words a "not"
# negator no longer forbids the verb, so "don't just delete the case, fix the parser" is rejected as well: it cannot
# be told apart from the additive form, and a rejection only falls back to the deterministic remedy.
_ADDITIVE_WORDS = frozenset({"only", "just", "merely", "simply", "solely", "exclusively", "purely"})
_NOT_NEGATORS = (*_LOOSE_NEGATORS, ("not",))
_OTHER_NEGATORS = (("without",), ("instead", "of"), ("rather", "than"), ("no", "need", "to"))
_MAX_MODIFIERS = 5
_MODIFIER_WORD = re.compile(r"just|also|then|even|ever|only|still|first|outright|again|altogether|always|\w{3,}ly")
_MODIFIER_PHRASES = frozenset(
    {("the", "need", "to"), ("any", "need", "to")}
    | {("being", participle, "to") for participle in ("forced", "asked", "required", "made")}
    | {
        (verb, "to")
        for verb in (
            "having",
            "have",
            "has",
            "needing",
            "need",
            "needs",
            "trying",
            "attempting",
            "resorting",
            "wanting",
            "choosing",
            "bothering",
        )
    }
)


def _ends_with(words: list[str], end: int, phrases: tuple[tuple[str, ...], ...]) -> bool:
    return any(end >= len(phrase) and tuple(words[end - len(phrase) : end]) == phrase for phrase in phrases)


def _modifier_start(words: list[str], end: int) -> int | None:
    """Start of the verb modifier ("first", "having to", "being forced to") that ends at ``end``, if any."""
    for size in (3, 2):
        if end >= size and tuple(words[end - size : end]) in _MODIFIER_PHRASES:
            return end - size
    if end >= 1 and _MODIFIER_WORD.fullmatch(words[end - 1]):
        return end - 1
    return None


def negated_before(text: str, start: int) -> bool:
    """True when advice that begins at ``start`` is negated ("do not ...", "without first ...", "instead of ...")."""
    words = _NEGATION_WORD.findall(text[max(0, start - _NEGATION_WINDOW) : start].lower())
    count = len(words)
    for plain in range(3):
        end = count - plain
        if (
            end >= 1
            and all(_PLAIN_WORD.fullmatch(word) and word not in _ADDITIVE_WORDS for word in words[end:])
            and _ends_with(words, end, _LOOSE_NEGATORS)
        ):
            return True
    end = count
    additive = False
    for _ in range(_MAX_MODIFIERS + 1):
        if _ends_with(words, end, _OTHER_NEGATORS):
            return True
        if _ends_with(words, end, _NOT_NEGATORS):
            return not additive
        modifier = _modifier_start(words, end)
        if modifier is None:
            return False
        additive = additive or any(word in _ADDITIVE_WORDS for word in words[modifier:end])
        end = modifier
    return False


_STRONG_VERB = re.compile(
    r"remov(?:e[sd]?|ing)|delet(?:e[sd]?|ing)|drop(?:s|ped|ping)?|skip(?:s|ped|ping)?|disabl(?:e[sd]?|ing)|"
    r"exclud(?:e[sd]?|ing)|discard(?:s|ed|ing)?|omit(?:s|ted|ting)?|eliminat(?:e[sd]?|ing)|retir(?:e[sd]?|ing)"
)
# "ignore case" and "strip case" are also about letter case, so these two verbs only count with a determiner:
# "ignore the case for positive-001" removes a case, "ignore case when comparing" does not.
_WEAK_VERB = re.compile(r"ignor(?:e[sd]?|ing)|strip(?:s|ped|ping)?")
# "take the case out of the dataset", "filter positive-001 out of evals.json": the object sits between verb and "out of".
_EXTRACTION_VERB = re.compile(
    r"tak(?:e[sn]?|ing)|took|leav(?:e|es|ing)|left|filter(?:s|ed|ing)?|pull(?:s|ed|ing)?|cut(?:s|ting)?"
)
_REPLACEMENT_VERB = re.compile(r"replac(?:e[sd]?|ing)")
_PHRASAL_VERBS = (
    (re.compile(r"comment(?:s|ed|ing)?"), ("out",)),
    (re.compile(r"stop(?:s|ped|ping)?"), ("running",)),
    (re.compile(r"get(?:s|ting)?|got|gotten"), ("rid", "of")),
    (re.compile(r"tak(?:e[sn]?|ing)|took"), ("out",)),
    (re.compile(r"leav(?:e|es|ing)|left"), ("out",)),
    (re.compile(r"filter(?:s|ed|ing)?"), ("out",)),
    (re.compile(r"swap(?:s|ped|ping)?"), ("out",)),
    (re.compile(r"instead"), ("of", "keeping")),
)
_REMOVE, _EXTRACT, _REPLACE = "remove", "extract", "replace"

_DETERMINER = re.compile(r"the|a|an|my|your|our|their|its|this|that|these|those|any|each|every|all|both")
_POSSESSIVE = re.compile(r"[a-z0-9_-]+(?:'s|\u2019s)")
# Words that end a noun phrase, so "remove temp files per case" is not read as "remove the case".
_PHRASE_STOP = re.compile(
    r"(?:in|of|on|at|by|for|to|if|as|and|or|with|without|from|after|before|when|where|that|which|than|then|but|so|"
    r"while|is|are|was|were|be|per|during|across|within|inside|into|onto|via|using|until|upon|around)\b"
)
_QUALIFIER_TOKEN = re.compile(r"[a-z0-9_'-]+")

_CASE_WORDS = frozenset({"case", "cases"})
_ENTRY_WORDS = frozenset({"entry", "entries", "row", "rows", "example", "examples"})
_QUALIFIERS = frozenset({"test", "eval", "evaluation", "dataset", "benchmark"})
_QUALIFIED_NOUN = re.compile(r"(?:test|eval(?:uation)?|dataset|benchmark)[_-]*(?:cases?|entr(?:y|ies)|rows?|examples?)")
_FAILURE_WORDS = frozenset({"mismatch", "mismatches", "failure", "failures", "violation", "violations"})
_FAILURE_LEAD = frozenset({"output", "answer", "result"})
# "case of", "case statement", "case folding", "case-insensitive" ...: letter case or control flow, not a dataset case.
_CASE_IDIOM = re.compile(r"of|statements?|conversions?|folding|handling|(?:in)?sensitiv.*")

_LINK_WORDS = frozenset({"for", "of", "named", "called", "titled", "labelled", "labeled", "id", "name"})
# An id that is not a known case id must end its phrase, so "the case for zero-length input" stays implementation prose.
_NAME_END = re.compile(
    r"so|to|and|or|but|because|since|as|which|that|if|when|by|then|from|in|within|out|"
    r"is|are|was|were|be|should|must|can|could|may|might|will|would"
)
_NAME_TEXT = re.compile(r"[a-z0-9_-]+")
_NAME_SEPARATOR = re.compile(r"[_-]")

_DATASET_DETERMINERS = frozenset({"the", "this", "our", "your"})
_EVAL_COMPONENT = re.compile(r"(?:^|/)evals?(?![a-z])")
_DATASET_WORD = re.compile(r"datasets?\b")
_JSON_FILE = re.compile(r"[\w./-]\.jsonl?\b")
_BENCHMARK = re.compile(r"benchmark\b")
# A case that sits in a script or source file ("the special case for positive-001 in parser.py") is an
# implementation branch, not a dataset row.
_CODE_FILE = re.compile(r"[\w./-]*\.(?:py|js|mjs|cjs|ts|tsx|jsx|sh|bash|rb|go|rs|java|c|cc|cpp|h|hpp|md)\b")
_CODE_WORD = re.compile(
    r"(?:script|code|function|parser|module|implementation|handler|logic|class|method|helper|switch|conditional|"
    r"branch|loop|program|algorithm|routine)\b"
)

_EASIER_WORD = re.compile(r"(?:easier|simpler|weaker|softer|looser|lighter|trivial)\b")
_BE_FORMS = frozenset({"is", "are", "was", "were", "be", "been", "being"})
_MODALS = frozenset({"should", "must", "can", "could", "may", "might", "will", "would"})

# What a case phrase is: a dataset case outright, a bare "case" that needs the clause to mention the dataset, or a
# known case id on its own (only enough for "replace positive-001 with an easier case").
_DATASET, _BARE, _ID_ONLY = "dataset", "bare", "id"


def _is_dataset_token(token: str) -> bool:
    return bool(
        _EVAL_COMPONENT.search(token)
        or _DATASET_WORD.search(token)
        or _JSON_FILE.search(token)
        or _BENCHMARK.match(token)
    )


def _id_shaped(token: str) -> bool:
    """True for case-id-like tokens: "positive-001", "syllable-count", "case_12", "python3"."""
    if not _NAME_TEXT.fullmatch(token):
        return False
    if any(char.isdigit() for char in token) and any("a" <= char <= "z" for char in token):
        return True
    parts = _NAME_SEPARATOR.split(token)
    if not all(parts):
        return False
    return len(parts) >= 2 or parts[0] in {"positive", "negative"}


def _clauses(text: str) -> Iterator[str]:
    start = 0
    for match in _CLAUSE_BREAK.finditer(text):
        yield text[start : match.end()]
        start = match.end()
    yield text[start:]


def _tokenize(clause: str) -> tuple[list[str], list[int], list[int]]:
    tokens: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for match in _TOKEN.finditer(clause):
        token = match.group()
        end = match.end()
        if _WORD_TOKEN.fullmatch(token):
            stripped = token.rstrip(".'\u2019/")
            if stripped != token:
                end -= len(token) - len(stripped)
                tokens.append(stripped)
                starts.append(match.start())
                ends.append(end)
                if token[len(stripped)] == ".":
                    tokens.append(".")
                    starts.append(end)
                    ends.append(match.end())
                continue
        tokens.append(token)
        starts.append(match.start())
        ends.append(end)
    return tokens, starts, ends


class _Clause:
    """One clause as tokens, with the lookups the removal checks share."""

    def __init__(self, text: str, known: frozenset[str], strict: bool) -> None:
        self.text = text
        self.known = known
        self.toks, self.starts, self.ends = _tokenize(text)
        self.n = len(self.toks)
        # Where each dataset location ("from evals.json", "out of the dataset") starts and where it ends.
        self.locations = {p: e for p in range(self.n) if (e := self._location_end(p)) is not None}
        self.first_location_end = min(self.locations.values(), default=None)
        # Identifier-shaped names are only evidence of a dataset case when the caller has no case ids to compare
        # against, or when the clause also points at the dataset: "the special case for UTF-8" is code.
        self.shape_counts = not strict or bool(self.locations)
        self.verbs_ending_at: dict[int, list[tuple[int, str, bool]]] = {}
        self.verbs_starting_at: dict[int, list[tuple[int, str, bool]]] = {}
        for i in range(self.n):
            for end, form, weak in self._verbs_at(i):
                self.verbs_starting_at.setdefault(i, []).append((end, form, weak))
                self.verbs_ending_at.setdefault(end - 1, []).append((i, form, weak))

    def removes(self) -> bool:
        return any(self._removal(start, end, kind) for start, end, kind in self._phrases())

    # --- tokens

    def _tok(self, index: int) -> str:
        return self.toks[index] if 0 <= index < self.n else ""

    def _verbs_at(self, i: int) -> Iterator[tuple[int, str, bool]]:
        token = self.toks[i]
        if _STRONG_VERB.fullmatch(token):
            yield i + 1, _REMOVE, False
        if _WEAK_VERB.fullmatch(token):
            yield i + 1, _REMOVE, True
        if _EXTRACTION_VERB.fullmatch(token):
            yield i + 1, _EXTRACT, False
        if _REPLACEMENT_VERB.fullmatch(token):
            yield i + 1, _REPLACE, False
        for first, rest in _PHRASAL_VERBS:
            if first.fullmatch(token) and tuple(self.toks[i + 1 : i + 1 + len(rest)]) == rest:
                yield i + 1 + len(rest), _REMOVE, False

    def _is_name(self, token: str) -> bool:
        return token in self.known or (self.shape_counts and _id_shaped(token))

    def _is_known_name(self, token: str) -> bool:
        return token in self.known

    def _strip_possessive(self, token: str) -> str:
        return token[:-2] if token.endswith(("'s", "\u2019s")) else token

    # --- dataset locations

    def _target_end(self, i: int) -> int | None:
        if self._tok(i) in _DATASET_DETERMINERS:
            i += 1
        token = self._tok(i)
        if token and _is_dataset_token(token):
            return i + 1
        if token in {"test", "eval", "evaluation"} and self._tok(i + 1) in {"set", "suite", "file"}:
            return i + 2
        return None

    def _location_end(self, i: int) -> int | None:
        token = self.toks[i]
        if token in {"from", "in", "within", "of"}:
            return self._target_end(i + 1)
        if token == "out" and self._tok(i + 1) == "of":
            return self._target_end(i + 2)
        return None

    def _location_within(self, noun: int, limit: int) -> bool:
        """A dataset location starts within ``limit`` characters of the noun, in the same phrase."""
        end = self.ends[noun]
        for p in range(noun + 1, self.n):
            gap = self.text[end : self.starts[p]]
            if len(gap) > limit or re.search(r"[.;!?]", gap):
                return False
            if p in self.locations:
                return True
        return False

    def _in_code(self, end: int) -> bool:
        """The phrase that ends at ``end`` is followed by "in parser.py", "from the script", ..."""
        token = self._tok(end)
        if token in {"in", "from", "inside", "within"}:
            i = end + 1
        elif token == "out" and self._tok(end + 1) == "of":
            i = end + 2
        else:
            return False
        if self._tok(i) in _DATASET_DETERMINERS:
            i += 1
        token = self._tok(i)
        return bool(token) and bool(_CODE_FILE.match(token) or _CODE_WORD.match(token))

    # --- case phrases

    def _fixes_failure(self, end: int) -> bool:
        """Failure phrases ("the test case failure", "the case mismatch") name a failure to repair, not the case."""
        if self._tok(end) in _FAILURE_LEAD:
            end += 1
        return self._tok(end) in _FAILURE_WORDS

    def _is_bare_case(self, index: int) -> bool:
        following = self._tok(index + 1)
        return not self._fixes_failure(index + 1) and not _CASE_IDIOM.fullmatch(following)

    def _name_starts(self, link_end: int) -> Iterator[int]:
        """Where a case id may start after a noun: directly, after ":", or after link words like "for the"."""
        yield link_end
        if self._tok(link_end) == ":":
            yield link_end + 1
        index = link_end
        for _ in range(2):
            token = self._tok(index)
            if token in _LINK_WORDS:
                index += 1
            elif token == "with":
                index += 1
                if self._tok(index) == "the":
                    index += 1
                if self._tok(index) not in {"id", "name"}:
                    return
                index += 1
            else:
                return
            yield index
            if self._tok(index) == "the":
                yield index + 1

    def _chain_start(self, noun: int, is_name: Callable[[str], bool]) -> int | None:
        """Start of the list of names ("a, b and c's") that ends right before ``noun``, if there is one."""
        last = noun - 1
        if last < 0 or not is_name(self._strip_possessive(self.toks[last])):
            return None
        start = last
        while True:
            index = start - 1
            if self._tok(index) in {"and", "or"}:
                index -= 1
                if self._tok(index) == ",":
                    index -= 1
            elif self._tok(index) == ",":
                index -= 1
            else:
                return start
            if index < 0 or not is_name(self.toks[index]):
                return start
            start = index

    def _phrases(self) -> Iterator[tuple[int, int, str]]:
        """Every (start, end, kind) token span that reads as a case in the dataset."""
        for k, token in enumerate(self.toks):
            if token in _CASE_WORDS:
                yield from self._case_phrases(k)
            elif token in _ENTRY_WORDS:
                yield from self._entry_phrases(k)
            elif _QUALIFIED_NOUN.fullmatch(token) and not self._fixes_failure(k + 1):
                yield k, k + 1, _DATASET
            if token in self.known:
                if k + 1 in self.locations:
                    yield k, k + 1, _DATASET
                yield k, k + 1, _ID_ONLY

    def _qualified(self, noun: int) -> Iterator[tuple[int, int, str]]:
        if self._tok(noun - 1) in _QUALIFIERS and not self._fixes_failure(noun + 1):
            yield noun - 1, noun + 1, _DATASET

    def _named_after(self, noun: int, *, any_shape: bool) -> Iterator[tuple[int, int, str]]:
        """ "the case for positive-001", "case: positive-001", "test case positive-001"."""
        for name in self._name_starts(noun + 1):
            token = self._tok(name)
            if self._strip_possessive(token) in self.known:
                if not self._in_code(name + 1):
                    yield noun, name + 1, _DATASET
            elif any_shape and self.shape_counts and token and _id_shaped(token):
                end = name + 1
                if (end >= self.n or self.toks[end] in _PHRASE_END_TOKENS or _NAME_END.fullmatch(self.toks[end])) and (
                    not self._in_code(end)
                ):
                    yield noun, end, _DATASET

    def _case_phrases(self, k: int) -> Iterator[tuple[int, int, str]]:
        yield from self._qualified(k)
        yield from self._named_after(k, any_shape=True)
        if self._is_bare_case(k):
            chain = self._chain_start(k, self._is_name)
            if chain is not None and not self._in_code(k + 1):
                yield chain, k + 1, _DATASET
            if self._location_within(k, 40):
                yield k, k + 1, _DATASET
            yield k, k + 1, _BARE

    def _entry_phrases(self, k: int) -> Iterator[tuple[int, int, str]]:
        yield from self._qualified(k)
        yield from self._named_after(k, any_shape=False)
        if self._location_within(k, 60):
            yield k, k + 1, _DATASET
        chain = self._chain_start(k, self._is_known_name)
        if chain is not None and not self._fixes_failure(k + 1) and not self._in_code(k + 1):
            yield chain, k + 1, _DATASET

    # --- removal verbs

    def _filler_ok(self, start: int, end: int, weak: bool) -> bool:
        """Between a verb and its case phrase: up to three determiners, then up to three plain words."""
        count = end - start
        if count > 6:
            return False
        determiners = 0
        while determiners < min(count, 3) and self._is_determiner(self.toks[start + determiners]):
            determiners += 1
        if weak and determiners == 0:
            return False
        rest = self.toks[start + determiners : end]
        return len(rest) <= 3 and all(
            _QUALIFIER_TOKEN.fullmatch(token) and not _PHRASE_STOP.match(token) for token in rest
        )

    def _is_determiner(self, token: str) -> bool:
        return bool(_DETERMINER.fullmatch(token) or _POSSESSIVE.fullmatch(token))

    def _scope_before(self, index: int) -> bool:
        return self.first_location_end is not None and self.first_location_end <= index

    def _easier_follows(self, index: int) -> bool:
        if self._tok(index) in {"a", "an", "another", "some"}:
            index += 1
        token = self._tok(index)
        if _EASIER_WORD.match(token):
            return True
        following = self._tok(index + 1)
        return (token == "more" and following in {"lenient", "forgiving", "permissive"}) or (
            token == "less" and following in {"strict", "demanding", "difficult", "challenging"}
        )

    def _replacement_follows(self, end: int) -> bool:
        for index in (end, self.locations.get(end)):
            if index is not None and self._tok(index) in {"with", "by"} and self._easier_follows(index + 1):
                return True
        return False

    def _suffix_ok(self, form: str, end: int, kind: str) -> bool:
        if form == _REPLACE:
            return self._replacement_follows(end)
        if kind == _ID_ONLY:
            return False
        if form == _EXTRACT:
            return self._tok(end) == "out" and self._tok(end + 1) == "of" and self._target_end(end + 2) is not None
        return True

    def _removal(self, start: int, end: int, kind: str) -> bool:
        return self._active(start, end, kind) or self._passive(start, end, kind)

    def _active(self, start: int, end: int, kind: str) -> bool:
        for gap in range(7):
            last = start - 1 - gap
            if last < 0:
                return False
            for verb_start, form, weak in self.verbs_ending_at.get(last, ()):
                if (
                    self._filler_ok(last + 1, start, weak)
                    and self._suffix_ok(form, end, kind)
                    and (kind != _BARE or self._scope_before(verb_start))
                    and not negated_before(self.text, self.starts[verb_start])
                ):
                    return True
        return False

    # --- passive removals: "the case should be excluded"

    def _passive_verb_at(self, index: int) -> bool:
        if any(form == _REMOVE for _, form, _ in self.verbs_starting_at.get(index, ())):
            return True
        return (
            self._tok(index) == "replaced"
            and self._tok(index + 1) in {"with", "by"}
            and self._easier_follows(index + 2)
        )

    def _tail_at(self, index: int) -> bool:
        if self._tok(index) in {"also", "then"}:
            index += 1
        token = self._tok(index)
        if token in _BE_FORMS:
            index += 1
        elif token in _MODALS and self._tok(index + 1) == "be":
            index += 2
        elif token in {"ought", "need", "needs"} and self.toks[index + 1 : index + 3] == ["to", "be"]:
            index += 3
        else:
            return False
        if self._tok(index) in {"just", "simply", "safely"}:
            index += 1
        return self._passive_verb_at(index)

    def _passive(self, start: int, end: int, kind: str) -> bool:
        if kind == _ID_ONLY or (kind == _BARE and not self._scope_before(start)):
            return False
        if negated_before(self.text, self.starts[start]):
            return False
        return self._tail_at(end) or (end in self.locations and self._tail_at(self.locations[end]))


def removes_dataset_case(text: str, case_ids: Iterable[object] = ()) -> bool:
    """True when ``text`` advises taking a dataset case out of the dataset (deleting, excluding, replacing it).

    ``case_ids`` are the ids of the dataset cases the advice may name. A known id is recognised next to "case" even
    when it is a plain word ("delete the smoke case"). Without ids, or when the clause points at the dataset
    ("... from evals.json"), identifier-shaped names ("positive-001", "syllable-count") are recognised too.
    """
    lowered = text.lower()
    supplied = [str(case_id).strip().lower() for case_id in case_ids]
    present = {name for name in supplied if len(name) >= 2 and name in lowered}
    # Ids that are not a single token (they contain spaces or other punctuation) are swapped for a placeholder.
    odd = sorted((name for name in present if not _WORD_TOKEN.fullmatch(name) or name[-1] in ".'\u2019/"), key=len)
    known = present.difference(odd)
    if odd:
        placeholders = {name: f"zzcaseid{index}zz" for index, name in enumerate(odd)}
        pattern = re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(n) for n in reversed(odd)) + r")(?![\w-])")
        lowered = pattern.sub(lambda match: placeholders[match.group()], lowered)
        known |= set(placeholders.values())
    strict = any(supplied)
    return any(_Clause(clause, frozenset(known), strict).removes() for clause in _clauses(lowered))
