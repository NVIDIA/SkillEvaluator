# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only validation of public and access-controlled GitHub Pages sites.

Repository visibility and Pages visibility are independent. Never create a site
for a nonpublic repository here: GitHub's documented REST create/update APIs do
not accept a Pages visibility setting. An administrator first selects Private
and GitHub Actions in Settings > Pages, then onboarding can verify and reuse it.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from skillevaluator._ci_onboarding_git import SetupError


def is_nonpublic(info: dict) -> bool:
    """Treat missing, inconsistent, and non-boolean privacy data conservatively."""
    return info.get("private") is not False or info.get("visibility", "public") != "public"


def _https_url(value: str) -> tuple[str, str]:
    """Validate an unambiguous HTTPS URL before using it in generated workflows."""
    if not isinstance(value, str) or not value or any(ord(char) <= 32 or ord(char) >= 127 for char in value):
        raise SetupError("The report URL must be an unambiguous HTTPS GitHub Pages URL.")
    try:
        parts = urlsplit(value)
        invalid = (
            parts.scheme != "https"
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.port is not None
            or parts.netloc.endswith(":")
            or parts.query
            or parts.fragment
            or "?" in value
            or "#" in value
            or "\\" in value
        )
    except ValueError as exc:
        raise SetupError("The report URL must be an unambiguous HTTPS GitHub Pages URL.") from exc
    if invalid:
        raise SetupError("The report URL must be an unambiguous HTTPS GitHub Pages URL.")
    return parts.hostname, parts.path


def _private_url(value: str) -> str:
    hostname, path = _https_url(value)
    labels = hostname.split(".")
    if (
        not hostname.endswith(".pages.github.io")
        or len(labels) < 4
        or len(hostname) > 253
        or not all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in labels)
        or path not in {"", "/"}
    ):
        raise SetupError("The private report URL must be GitHub's HTTPS *.pages.github.io site root.")
    return f"https://{hostname}/"


def report_url(name: str, custom: str | None = None, *, verified_private_url: str | None = None) -> str:
    """Accept a default public URL, or the exact private URL verified by GitHub.

    ``verified_private_url`` must come from ``private_pages_url`` using an
    authenticated GET of this repository's Pages resource, never user input.
    Supplying a private hostname alone does not establish its repository owner.
    """
    owner, repo_name = name.split("/")
    expected = (
        _private_url(verified_private_url)
        if verified_private_url is not None
        else f"https://{owner.lower()}.github.io/{repo_name}/"
    )
    value = expected if custom is None else custom
    _https_url(value)
    if verified_private_url is not None:
        value = _private_url(value)
    if value.removesuffix("/") != expected.removesuffix("/"):
        raise SetupError("The report URL must match this repository's verified GitHub Pages URL.")
    return expected


def check_private_repository(name: str, info: dict) -> None:
    """Reject unsupported owners and uncertain permissions before querying Pages."""
    owner, repo_name = name.split("/")
    if info.get("private") is not True or info.get("visibility") not in {"private", "internal"}:
        raise SetupError(
            "GitHub did not confirm this repository is private or internal; inspect its visibility or use --checks-only."
        )
    if not isinstance(info.get("owner"), dict) or info["owner"].get("type") != "Organization":
        raise SetupError(
            "Private Pages requires a private or internal project repository owned by a GitHub Enterprise Cloud "
            "organization. Personal repositories cannot publish private Pages. Use --checks-only to keep reports "
            "in Actions artifacts."
        )
    if repo_name.lower() == f"{owner.lower()}.github.io":
        raise SetupError(
            "Private Pages access control is unavailable for an organization site. Use an organization project "
            "repository, or --checks-only to keep reports in Actions artifacts."
        )
    if not isinstance(info.get("permissions"), dict) or info["permissions"].get("admin") is not True:
        raise SetupError(
            "Private Pages setup requires repository admin access and a token with Pages and Administration "
            "permissions. Ask an owner to configure the site, or use --checks-only."
        )


def private_pages_url(name: str, info: dict, pages: dict | None) -> str:
    """Return the verified private origin or explain the prerequisite to fix.

    The caller owns API reads and existing history verification. This function
    makes no request and does not infer access control from repository privacy,
    URL shape, a missing resource, or a plan name.
    """
    check_private_repository(name, info)
    settings = f"https://github.com/{name}/settings/pages"
    if not isinstance(pages, dict) or pages.get("public") is not False:
        raise SetupError(
            "GitHub has not verified a private Pages site for this repository. Open "
            f"{settings}, select Private visibility and GitHub Actions as the source, then rerun setup. "
            "Private visibility requires GitHub Enterprise Cloud and organization policy that allows private Pages. "
            "If the option is unavailable, ask an organization owner or use --checks-only. "
            "A public site will not be created."
        )
    if pages.get("build_type") not in {"workflow", "legacy"} or pages.get("cname"):
        raise SetupError(
            "The existing private Pages configuration is outside SkillEvaluator's setup. "
            "Use the default GitHub private domain and GitHub Actions source, or use --checks-only."
        )
    return _private_url(pages.get("html_url"))


def check_site(name: str, pages: dict | None, *, history_exists: bool, deployments: list | None = None) -> None:
    """Protect existing content while allowing a fresh private Actions site.

    ``history_exists`` must be the result of the caller's ownership check: False
    means no gh-pages branch exists, not an existing unowned or invalid branch.
    For adoption, ``deployments`` must come from an authenticated, unfiltered
    listing of this repository's deployments. Pages can use any environment,
    so checking only github-pages cannot establish emptiness. None means not
    verified.
    """
    if pages is None:
        return
    if not isinstance(pages, dict) or pages.get("build_type") not in {"workflow", "legacy"} or pages.get("cname"):
        raise SetupError("An existing Pages site is outside SkillEvaluator's setup. It will not be replaced.")
    if history_exists:
        return
    if (
        pages.get("public") is False
        and pages.get("build_type") == "workflow"
        and "status" in pages
        and pages["status"] is None
        and isinstance(deployments, list)
        and not deployments
    ):
        return
    raise SetupError(
        "An existing Pages site is not owned by SkillEvaluator and is not verified as a fresh empty private site. "
        "Its content will not be replaced. Use --checks-only or a fresh private Pages project site at "
        f"https://github.com/{name}/settings/pages."
    )
