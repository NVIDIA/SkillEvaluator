# SPDX-License-Identifier: Apache-2.0
"""Bounded authenticated GitLab GraphQL relationship reads."""

import http.client
import io
import json
import urllib.error

import pytest

from ci import feedback


class Response(io.BytesIO):
    def __init__(self, body, status=200):
        super().__init__(body)
        self.status = status
        self.headers = {}

    def getcode(self):
        return self.status


class Opener:
    def __init__(self, body, status=200, error=None):
        self.body, self.status, self.error = body, status, error
        self.requests = []

    def open(self, request, *, timeout):
        assert timeout > 0
        self.requests.append(request)
        if self.error:
            raise self.error
        return Response(self.body, self.status)


def test_graphql_uses_only_the_configured_instance_origin_and_json_post():
    api = feedback.GitLabAPI("https://gitlab.example/company/api/v4", "12", "private-feedback-token")
    api.opener = Opener(b'{"data":{"project":{"fullPath":"acme/skills"}}}')
    data = api.graphql("query($project:ID!){project(fullPath:$project){fullPath}}", {"project": "acme/skills"})
    assert data == {"project": {"fullPath": "acme/skills"}}
    request = api.opener.requests[0]
    assert request.full_url == "https://gitlab.example/company/api/graphql"
    assert request.method == "POST"
    assert request.get_header("Private-token") == "private-feedback-token"
    assert json.loads(request.data)["variables"] == {"project": "acme/skills"}


@pytest.mark.parametrize("status", [301, 302, 307, 400, 401, 403, 500])
def test_graphql_http_errors_and_redirects_never_forward_or_echo_credentials(status):
    api = feedback.GitLabAPI("https://gitlab.example/api/v4", "12", "private-feedback-token")
    error = urllib.error.HTTPError(
        "https://gitlab.example/api/graphql",
        status,
        "private-feedback-token",
        {"Location": "https://evil.example"},
        None,
    )
    api.opener = Opener(b"", error=error)
    with pytest.raises(feedback.FeedbackError) as raised:
        api.graphql("query{__typename}", {})
    assert "private-feedback-token" not in str(raised.value)
    assert len(api.opener.requests) == 1
    assert api.opener.requests[0].full_url == "https://gitlab.example/api/graphql"


@pytest.mark.parametrize(
    "error", [http.client.BadStatusLine("private-feedback-token"), http.client.HTTPException("private-feedback-token")]
)
def test_graphql_sanitizes_low_level_http_parser_errors(error):
    api = feedback.GitLabAPI("https://gitlab.example/api/v4", "12", "private-feedback-token")
    api.opener = Opener(b"", error=error)
    with pytest.raises(feedback.FeedbackError) as raised:
        api.graphql("query{__typename}", {})
    assert "private-feedback-token" not in str(raised.value)
    assert raised.value.__cause__ is None


@pytest.mark.parametrize(
    "body",
    [
        b'{"errors":[{"message":"private-feedback-token"}],"data":{"project":{}}}',
        b'{"data":null}',
        b'{"data":{}} trailing',
        b'{"data":{},"data":{}}',
        b"[]",
    ],
)
def test_graphql_rejects_error_partial_or_malformed_responses(body):
    api = feedback.GitLabAPI("https://gitlab.example/api/v4", "12", "private-feedback-token")
    api.opener = Opener(body)
    with pytest.raises(feedback.FeedbackError) as raised:
        api.graphql("query{__typename}", {})
    assert "private-feedback-token" not in str(raised.value)


def test_graphql_response_and_request_sizes_are_bounded(monkeypatch):
    api = feedback.GitLabAPI("https://gitlab.example/api/v4", "12", "private-feedback-token")
    api.opener = Opener(b'{"data":{"value":"' + b"x" * 80 + b'"}}')
    monkeypatch.setattr(feedback.github_api, "MAX_API_BYTES", 64)
    with pytest.raises(feedback.FeedbackError, match="size limit"):
        api.graphql("query{__typename}", {})
    api.opener.requests.clear()
    with pytest.raises(feedback.FeedbackError):
        api.graphql("query{__typename}", {"project": "x" * (64 * 1024)})
    assert not api.opener.requests


def test_graphql_rejects_a_non_error_http_redirect_response():
    api = feedback.GitLabAPI("https://gitlab.example/api/v4", "12", "private-feedback-token")
    api.opener = Opener(b'{"data":{}}', status=302)
    with pytest.raises(feedback.FeedbackError):
        api.graphql("query{__typename}", {})
