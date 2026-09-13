# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unicode security regression coverage for public identity placeholders."""

from __future__ import annotations

import pytest

from skillevaluator import publication_text
from skillevaluator.publication_text import (
    UNICODE_CONFUSABLES_GENERATOR_UCD_VERSION,
    UNICODE_CONFUSABLES_SOURCE_SHA256,
    UNICODE_CONFUSABLES_SUBSET_SOURCE_COUNT,
    UNICODE_CONFUSABLES_VERSION,
    publication_identity_present,
)


def test_confusables_table_is_pinned_to_unicode_17() -> None:
    assert UNICODE_CONFUSABLES_VERSION == "17.0.0"
    assert UNICODE_CONFUSABLES_GENERATOR_UCD_VERSION == "15.1.0"
    assert UNICODE_CONFUSABLES_SUBSET_SOURCE_COUNT == 1954
    assert UNICODE_CONFUSABLES_SOURCE_SHA256 == "091c7f82fc39ef208faf8f94d29c244de99254675e09de163160c810d13ef22a"


@pytest.mark.parametrize(
    "identity",
    [
        "unkn0wn",
        "\u039codel not recorded",
        "model not re\u03f2orded",
        "unkn\u0c02wn",
        "unkn\U0001cce4wn",
        "mode\u0140not recorded",
        "model\u0149ot recorded",
        "unknow\u145amodel",
        "u\u0295nknown",
        "u\U0001f40dnknown",
        "u\u6138nknown",
    ],
    ids=[
        "digit",
        "uppercase",
        "lunate-sigma",
        "mark",
        "post-runtime-unicode",
        "trailing-separator",
        "leading-separator",
        "folded-trailing-separator",
        "vanishing-letter",
        "vanishing-symbol",
        "vanishing-cjk",
    ],
)
def test_pinned_confusable_substitutions_cannot_bypass_reserved_identities(identity: str) -> None:
    assert not publication_identity_present(identity)


@pytest.mark.parametrize(
    "identity", ["Caf\u00e9", "\u30a8\u30fc\u30b8\u30a7\u30f3\u30c8", "\u6a21\u578b", "\u0646\u0645\u0648\u0630\u062c"]
)
def test_recorded_multilingual_identities_remain_valid(identity: str) -> None:
    assert publication_identity_present(identity)


def test_unicode_15_1_identity_is_stable_on_a_unicode_15_0_host(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = "\U0002ebf0"
    host_category = publication_text.unicodedata.category
    monkeypatch.setattr(
        publication_text.unicodedata,
        "category",
        lambda character: "Cn" if character == identity else host_category(character),
    )

    assert publication_text.publication_semantic_text(identity) == identity
    assert publication_identity_present(identity)
