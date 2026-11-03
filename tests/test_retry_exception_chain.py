# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retain provider status and certificate evidence through exception wrappers."""

from __future__ import annotations

import ssl
import urllib.error

import httpx
import pytest

from skillevaluator.inference.retry import extract_http_status, is_retriable_exception


def _status(code):
    return urllib.error.HTTPError("http://localhost/", code, "test", {}, None)


@pytest.mark.parametrize("link", ["__cause__", "__context__"])
def test_status_comes_from_innermost_link_with_an_http_status(link):
    wrapper = _status(500)
    intermediate = RuntimeError("provider wrapper")
    underlying = httpx.HTTPStatusError(
        "real provider response",
        request=httpx.Request("POST", "http://localhost/"),
        response=httpx.Response(505),
    )
    setattr(wrapper, link, intermediate)
    setattr(intermediate, link, underlying)
    assert extract_http_status(wrapper) == 505
    assert not is_retriable_exception(wrapper)


def test_explicit_cause_wins_over_unrelated_implicit_context():
    wrapper = _status(500)
    wrapper.__context__ = _status(501)
    wrapper.__cause__ = _status(429)
    assert extract_http_status(wrapper) == 429
    assert is_retriable_exception(wrapper)


def test_suppressed_context_does_not_replace_the_current_failure():
    wrapper = _status(503)
    wrapper.__context__ = urllib.error.URLError(ssl.SSLCertVerificationError("previous failure"))
    wrapper.__suppress_context__ = True
    assert extract_http_status(wrapper) == 503
    assert is_retriable_exception(wrapper)


def test_cyclic_chain_keeps_innermost_distinct_status_without_looping():
    wrapper, inner = _status(500), _status(501)
    wrapper.__cause__ = inner
    inner.__context__ = wrapper
    assert extract_http_status(wrapper) == 501
    assert not is_retriable_exception(wrapper)


def test_certificate_evidence_survives_more_than_32_wrappers_and_a_cycle():
    certificate = ssl.SSLCertVerificationError("untrusted certificate")
    current = certificate
    for _ in range(40):
        wrapper = RuntimeError("wrapper")
        wrapper.__cause__ = current
        current = wrapper
    outer = _status(503)
    outer.__cause__ = current
    certificate.__context__ = outer
    assert not is_retriable_exception(outer)


def test_cycle_without_certificate_keeps_connection_failure_retriable():
    error = ConnectionResetError("reset")
    error.__cause__ = error
    assert is_retriable_exception(error)


def test_status_without_http_response_keeps_wrapper_status():
    outer = _status(502)
    outer.__cause__ = OSError("connection reset")
    assert extract_http_status(outer) == 502
    assert is_retriable_exception(outer)
