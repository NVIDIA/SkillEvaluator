# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded one-task Harbor smoke runs for agent runtime readiness."""

from __future__ import annotations

import errno
import json
import math
import os
import shlex
import ssl
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from queue import Empty, Queue
from threading import BoundedSemaphore, Thread
from time import monotonic
from typing import TYPE_CHECKING
from urllib.parse import urlsplit
from urllib.request import getproxies

import boto3
from botocore import exceptions as botocore_exceptions
from botocore.config import Config as BotoConfig
from botocore.exceptions import (
    ApiVersionNotFoundError,
    BaseEndpointResolverError,
    BotoCoreError,
    ClientError,
    ConfigNotFound,
    ConfigParseError,
    CredentialRetrievalError,
    DataNotFoundError,
    EndpointProviderError,
    HTTPClientError,
    InvalidConfigError,
    InvalidDefaultsMode,
    InvalidIMDSEndpointError,
    InvalidIMDSEndpointModeError,
    InvalidMaxRetryAttemptsError,
    InvalidProxiesConfigError,
    InvalidRegionError,
    InvalidRetryConfigurationError,
    InvalidSTSRegionalEndpointsConfigError,
    MissingDependencyException,
    NoAuthTokenError,
    NoCredentialsError,
    ParamValidationError,
    PartialCredentialsError,
    ProfileNotFound,
    RefreshWithMFAUnsupportedError,
    ServiceNotInRegionError,
    SSOTokenLoadError,
    UnauthorizedSSOTokenError,
    UnknownCredentialError,
    UnknownRegionError,
    UnknownSignatureVersionError,
    UnsupportedSignatureVersionError,
)
from botocore.exceptions import (
    SSLError as BotocoreSSLError,
)

from skillevaluator.model_catalog import (
    ModelCatalogError,
    ModelCatalogFailureKind,
    fetch_anthropic_model_record,
    fetch_model_records,
)
from skillevaluator.tier3.harbor.progress import redact_progress_detail
from skillevaluator.tier3.harbor.runner import _nvidia_build_key_handoff, build_harbor_run_command

if TYPE_CHECKING:
    from skillevaluator.provider_config import ProviderConfig

DEFAULT_PREFLIGHT_TIMEOUT_SECONDS = 900


@dataclass(frozen=True)
class PreflightResult:
    """Persistable outcome of a real, verification-disabled agent smoke."""

    ok: bool
    agent: str
    model: str
    detail: str
    job_name: str


@dataclass(frozen=True)
class ModelProbeResult:
    """Safe result from a provider model-catalog request."""

    ok: bool
    provider: str
    model: str
    detail: str
    failure_kind: ModelCatalogFailureKind | None = None
    http_status: int | None = None
    catalog_authoritative: bool = True


class CredentialProbeDisposition(StrEnum):
    """How Tier 3 should act on a model-catalog credential probe."""

    VERIFIED = "verified"
    FATAL = "fatal"
    DEGRADED = "degraded"


_BEDROCK_PROBE_SLOT = BoundedSemaphore(4)
_BEDROCK_AUTHENTICATION_ERRORS: tuple[type[BaseException], ...] = (
    NoAuthTokenError,
    NoCredentialsError,
    RefreshWithMFAUnsupportedError,
    SSOTokenLoadError,
    UnauthorizedSSOTokenError,
)
_BEDROCK_INVALID_CONFIGURATION_ERRORS: tuple[type[BaseException], ...] = (
    ApiVersionNotFoundError,
    BaseEndpointResolverError,
    ConfigNotFound,
    ConfigParseError,
    DataNotFoundError,
    EndpointProviderError,
    InvalidConfigError,
    InvalidDefaultsMode,
    InvalidIMDSEndpointError,
    InvalidIMDSEndpointModeError,
    InvalidProxiesConfigError,
    InvalidRegionError,
    InvalidRetryConfigurationError,
    InvalidSTSRegionalEndpointsConfigError,
    MissingDependencyException,
    ParamValidationError,
    PartialCredentialsError,
    ProfileNotFound,
    ServiceNotInRegionError,
    UnknownCredentialError,
    UnknownRegionError,
    UnknownSignatureVersionError,
    UnsupportedSignatureVersionError,
)
for _error_name, _target in (
    ("LoginError", "authentication"),
    ("InvalidChecksumConfigError", "configuration"),
    ("UnknownTokenProviderError", "configuration"),
    ("UnsupportedServiceProtocolsError", "configuration"),
):
    _error_type = getattr(botocore_exceptions, _error_name, None)
    if not isinstance(_error_type, type) or not issubclass(_error_type, BotoCoreError):
        continue
    if _target == "authentication":
        _BEDROCK_AUTHENTICATION_ERRORS += (_error_type,)
    else:
        _BEDROCK_INVALID_CONFIGURATION_ERRORS += (_error_type,)

_BEDROCK_CREDENTIAL_BOOTSTRAP_TRANSIENT_CODES = frozenset(
    {
        "IDPCommunicationError",
        "IDPCommunicationErrorException",
        "InternalFailure",
        "InternalServerError",
        "InternalServerException",
        "RequestTimeout",
        "RequestTimeoutException",
        "ServiceUnavailable",
        "ServiceUnavailableException",
        "Throttling",
        "ThrottlingException",
        "TooManyRequestsError",
        "TooManyRequestsException",
    }
)

_BEDROCK_INTEGER_ENVIRONMENT_VARIABLES = (
    "AWS_MAX_ATTEMPTS",
    "AWS_METADATA_SERVICE_NUM_ATTEMPTS",
    "AWS_METADATA_SERVICE_TIMEOUT",
)
_BEDROCK_ENDPOINT_ENVIRONMENT_VARIABLES = (
    "AWS_ENDPOINT_URL",
    "AWS_ENDPOINT_URL_BEDROCK",
    "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
    "AWS_ENDPOINT_URL_STS",
)
_BEDROCK_RUNTIME_SERVICE_MODELS = ("bedrock", "bedrock-runtime")
_BEDROCK_CREDENTIAL_PROCESS_TRANSIENT_ERRNOS = frozenset(
    {errno.EAGAIN, errno.EINTR, errno.EMFILE, errno.ENFILE, errno.ENOMEM, errno.ETIMEDOUT}
)


def _bedrock_scoped_config(session: object | None) -> Mapping[str, object]:
    botocore_session = getattr(session, "_session", None)
    get_scoped_config = getattr(botocore_session, "get_scoped_config", None)
    if not callable(get_scoped_config):
        return {}
    try:
        scoped_config = get_scoped_config()
    except Exception:
        return {}
    if not isinstance(scoped_config, Mapping):
        return {}
    return scoped_config


def _bedrock_credential_process(session: object | None) -> str | None:
    credential_process = _bedrock_scoped_config(session).get("credential_process")
    return credential_process if isinstance(credential_process, str) else None


def _bedrock_active_credential_service_models(session: object | None) -> tuple[str, ...]:
    profile_configs = [_bedrock_scoped_config(session)]
    botocore_session = getattr(session, "_session", None)
    full_config = getattr(botocore_session, "full_config", None)
    profiles = full_config.get("profiles") if isinstance(full_config, Mapping) else None
    visited_profiles: set[str] = set()
    source_profile = profile_configs[0].get("source_profile")
    while isinstance(source_profile, str) and isinstance(profiles, Mapping) and source_profile not in visited_profiles:
        visited_profiles.add(source_profile)
        source_config = profiles.get(source_profile)
        if not isinstance(source_config, Mapping):
            break
        profile_configs.append(source_config)
        source_profile = source_config.get("source_profile")

    services: list[str] = []
    if any("role_arn" in config for config in profile_configs) or (
        "AWS_ROLE_ARN" in os.environ and "AWS_WEB_IDENTITY_TOKEN_FILE" in os.environ
    ):
        services.append("sts")
    if any(any(variable in config for variable in ("sso_start_url", "sso_session")) for config in profile_configs):
        services.extend(("sso", "sso-oidc"))
    if any("login_session" in config for config in profile_configs):
        services.append("signin")
    return tuple(services)


def _is_unusable_imds_credential_response(request: object, response: object) -> bool:
    if getattr(request, "method", None) != "GET" or getattr(response, "status_code", None) != 200:
        return False
    request_url = getattr(request, "url", None)
    if not isinstance(request_url, str):
        return False
    try:
        path = urlsplit(request_url).path
    except ValueError:
        return False
    credential_path = "/latest/meta-data/iam/security-credentials/"
    if not path.startswith(credential_path):
        return False
    try:
        body = response.text
    except Exception:
        return True
    if path.rstrip("/") == credential_path.rstrip("/"):
        allowed_role_characters = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_+=,.@-")
        return not (1 <= len(body) <= 64 and all(character in allowed_role_characters for character in body))
    if not body.strip():
        return True
    try:
        payload = json.loads(body)
    except (TypeError, ValueError):
        return True
    required_fields = {"AccessKeyId", "SecretAccessKey", "Token", "Expiration"}
    return not isinstance(payload, Mapping) or not required_fields.issubset(payload)


def _monitor_bedrock_metadata_unavailability(session: object) -> list[bool]:
    """Retain transient metadata failures that Botocore may collapse to no credentials."""
    unavailable = [False]
    botocore_session = getattr(session, "_session", None)
    get_component = getattr(botocore_session, "get_component", None)
    if not callable(get_component):
        return unavailable
    try:
        resolver = get_component("credential_provider")
        get_provider = resolver.get_provider
    except Exception:
        return unavailable

    transports: list[object] = []
    for provider_name, fetcher_name in (("container-role", "_fetcher"), ("iam-role", "_role_fetcher")):
        try:
            provider = get_provider(provider_name)
            fetcher = getattr(provider, fetcher_name)
            transport = fetcher._session
            if callable(getattr(transport, "send", None)):
                transports.append(transport)
        except Exception:
            continue

    for transport in transports:
        original_send = transport.send

        def tracked_send(request, *args, _send=original_send, **kwargs):
            try:
                response = _send(request, *args, **kwargs)
            except Exception:
                unavailable[0] = True
                raise
            status_code = getattr(response, "status_code", None)
            if (
                isinstance(status_code, int)
                and not isinstance(status_code, bool)
                and (status_code in {408, 429} or status_code >= 500)
            ) or _is_unusable_imds_credential_response(request, response):
                unavailable[0] = True
            return response

        try:
            transport.send = tracked_send
        except Exception:
            continue
    return unavailable


def _validate_bedrock_local_sdk_configuration(session: object) -> None:
    """Force local values that the probe's bounded overrides could mask."""
    botocore_session = getattr(session, "_session", None)
    get_config_variable = getattr(botocore_session, "get_config_variable", None)
    if callable(get_config_variable):
        max_attempts = get_config_variable("max_attempts")
        if max_attempts is not None and max_attempts < 1:
            raise InvalidMaxRetryAttemptsError(provided_max_attempts=max_attempts, min_value=1)
        get_config_variable("metadata_service_num_attempts")
        metadata_service_timeout = get_config_variable("metadata_service_timeout")
        if metadata_service_timeout is not None and metadata_service_timeout <= 0:
            raise InvalidConfigError(error_msg="metadata service timeout must be greater than zero")
        if get_config_variable("csm_enabled"):
            get_config_variable("csm_port")
    get_service_model = getattr(botocore_session, "get_service_model", None)
    if callable(get_service_model):
        for service_name in _BEDROCK_RUNTIME_SERVICE_MODELS:
            get_service_model(service_name)


def _is_invalid_bedrock_ca_bundle(session: object | None) -> bool:
    botocore_session = getattr(session, "_session", None)
    get_config_variable = getattr(botocore_session, "get_config_variable", None)
    try:
        value = get_config_variable("ca_bundle") if callable(get_config_variable) else os.environ.get("AWS_CA_BUNDLE")
    except Exception:
        value = os.environ.get("AWS_CA_BUNDLE")
    if not isinstance(value, str) or not value:
        return False
    try:
        ssl.create_default_context(cafile=value)
    except OSError:
        return True
    return False


def _bedrock_client_uses_tls(client: object) -> bool:
    endpoint_url = getattr(getattr(client, "meta", None), "endpoint_url", None)
    return isinstance(endpoint_url, str) and urlsplit(endpoint_url).scheme.casefold() == "https"


def _is_builtin_botocore_data_path(loader: object, loaded_path: object) -> bool:
    builtin_data_path = getattr(loader, "BUILTIN_DATA_PATH", None)
    if not isinstance(builtin_data_path, str) or not isinstance(loaded_path, str):
        return False
    try:
        builtin_root = Path(builtin_data_path).expanduser().resolve()
        loaded_base = Path(loaded_path).expanduser()
    except OSError:
        return False
    found_data_file = False
    for suffix in (".json", ".json.gz"):
        candidate = Path(f"{loaded_base}{suffix}")
        try:
            if not candidate.is_file():
                continue
            found_data_file = True
            resolved_path = candidate.resolve()
        except OSError:
            return False
        if resolved_path != builtin_root and builtin_root not in resolved_path.parents:
            return False
    return found_data_file


def _bedrock_endpoint_metadata_is_builtin(
    session: object,
    client: object,
    *,
    service_name: str,
) -> bool:
    """Require endpoint authority to come from Botocore's packaged data."""
    botocore_session = getattr(session, "_session", None)
    get_component = getattr(botocore_session, "get_component", None)
    if not callable(get_component):
        return False
    try:
        loader = get_component("data_loader")
        load_data_with_path = loader.load_data_with_path
        for data_name in ("endpoints", "partitions"):
            _data, loaded_path = load_data_with_path(data_name)
            if not _is_builtin_botocore_data_path(loader, loaded_path):
                return False

        service_model = getattr(getattr(client, "meta", None), "service_model", None)
        api_version = getattr(service_model, "api_version", None)
        if not isinstance(api_version, str) or not api_version:
            api_version = loader.determine_latest_version(service_name, "endpoint-rule-set-1")
        _rules, rules_path = load_data_with_path(f"{service_name}/{api_version}/endpoint-rule-set-1")
        return _is_builtin_botocore_data_path(loader, rules_path)
    except Exception:
        return False


def _is_native_bedrock_endpoint(
    session: object,
    client: object,
    *,
    service_name: str,
    region_name: str,
) -> bool:
    endpoint_url = getattr(getattr(client, "meta", None), "endpoint_url", None)
    if endpoint_url is None:
        # Lightweight test doubles predate endpoint-authority evidence.
        return True
    if not isinstance(endpoint_url, str):
        return False
    try:
        endpoint = urlsplit(endpoint_url)
        port = endpoint.port
    except ValueError:
        return False
    if (
        endpoint.scheme.casefold() != "https"
        or endpoint.hostname is None
        or endpoint.username is not None
        or endpoint.password is not None
        or port not in {None, 443}
        or endpoint.path not in {"", "/"}
        or endpoint.query
        or endpoint.fragment
    ):
        return False
    if not _bedrock_endpoint_metadata_is_builtin(session, client, service_name=service_name):
        return False

    botocore_session = getattr(session, "_session", None)
    get_internal_component = getattr(botocore_session, "_get_internal_component", None)
    if not callable(get_internal_component):
        return False
    try:
        resolved = get_internal_component("endpoint_resolver").construct_endpoint(service_name, region_name)
    except Exception:
        return False
    if not isinstance(resolved, Mapping):
        return False
    allowed_hosts = {str(resolved.get("hostname") or "").casefold()}
    variants = resolved.get("variants")
    if isinstance(variants, list):
        for variant in variants:
            if not isinstance(variant, Mapping):
                continue
            hostname = variant.get("hostname")
            if not isinstance(hostname, str):
                continue
            try:
                allowed_hosts.add(
                    hostname.format(
                        service=service_name,
                        region=region_name,
                        dnsSuffix=variant.get("dnsSuffix", resolved.get("dnsSuffix", "")),
                    ).casefold()
                )
            except (KeyError, ValueError):
                continue
    return endpoint.hostname.casefold() in allowed_hosts


def _is_bedrock_local_cached_token_error(exc: Exception) -> bool:
    """Identify date parsing failures sourced from local SSO/login caches."""
    error_type = type(exc)
    parser_error = error_type.__name__ == "ParserError" and error_type.__module__.startswith("dateutil.parser")
    if not parser_error and not isinstance(exc, TypeError):
        return False
    traceback = exc.__traceback__
    saw_dateutil_parser = parser_error
    saw_local_cache_owner = False
    while traceback is not None:
        frame = traceback.tb_frame
        module_name = frame.f_globals.get("__name__")
        owner = frame.f_locals.get("self")
        owner_name = type(owner).__name__ if owner is not None else ""
        if isinstance(module_name, str) and module_name.startswith("dateutil.parser"):
            saw_dateutil_parser = True
        if module_name == "botocore.credentials" and owner_name in {"LoginProvider", "SSOCredentialFetcher"}:
            saw_local_cache_owner = True
        if module_name == "botocore.tokens" and owner_name == "SSOTokenProvider":
            saw_local_cache_owner = True
        traceback = traceback.tb_next
    return saw_dateutil_parser and saw_local_cache_owner


def _is_bedrock_local_sdk_data_error(exc: Exception) -> bool:
    """Identify failures raised while Botocore reads local model files."""
    traceback = exc.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        module_name = frame.f_globals.get("__name__")
        if module_name == "botocore.loaders" and frame.f_code.co_name == "_load_file":
            return True
        traceback = traceback.tb_next
    return False


def _is_bedrock_local_credential_file_error(exc: Exception) -> bool:
    """Identify text-decoding failures from configured local credential files."""
    if not isinstance(exc, UnicodeError):
        return False
    traceback = exc.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        module_name = frame.f_globals.get("__name__")
        owner = frame.f_locals.get("self")
        owner_name = type(owner).__name__ if owner is not None else ""
        if (
            module_name == "botocore.utils"
            and frame.f_code.co_name == "__call__"
            and owner_name == "FileWebIdentityTokenLoader"
        ):
            return True
        if (
            module_name == "botocore.credentials"
            and frame.f_code.co_name == "_build_headers"
            and owner_name == "ContainerProvider"
        ):
            return True
        traceback = traceback.tb_next
    return False


def _is_bedrock_local_credential_process_output_error(exc: Exception) -> bool:
    """Identify malformed output decoded by a configured credential_process."""
    traceback = exc.__traceback__
    saw_process_provider_load = False
    saw_date_parser = False
    while traceback is not None:
        frame = traceback.tb_frame
        module_name = frame.f_globals.get("__name__")
        owner = frame.f_locals.get("self")
        if module_name == "botocore.credentials" and type(owner).__name__ == "ProcessProvider":
            if frame.f_code.co_name == "_retrieve_credentials_using":
                process = frame.f_locals.get("p")
                return_code = getattr(process, "returncode", None)
                return (
                    isinstance(return_code, int)
                    and return_code == 0
                    and isinstance(
                        exc,
                        (json.JSONDecodeError, UnicodeError, AttributeError, TypeError, CredentialRetrievalError),
                    )
                )
            if frame.f_code.co_name == "load":
                saw_process_provider_load = True
        if isinstance(module_name, str) and module_name.startswith("dateutil.parser"):
            saw_date_parser = True
        traceback = traceback.tb_next
    return saw_process_provider_load and saw_date_parser and isinstance(exc, (TypeError, ValueError))


def _is_bedrock_local_proxy_configuration_error(exc: Exception) -> bool:
    """Match a proxy parser failure to a locally configured proxy value."""
    if not isinstance(exc, HTTPClientError):
        return False
    underlying = exc.kwargs.get("error") if isinstance(exc.kwargs, Mapping) else None
    urllib3_parse_error = type(underlying).__name__ == "LocationParseError" and type(underlying).__module__.startswith(
        "urllib3.exceptions"
    )
    if not isinstance(underlying, ValueError) and not urllib3_parse_error:
        return False
    for proxy_kind, configured_proxy in getproxies().items():
        if proxy_kind == "no" or not isinstance(configured_proxy, str) or not configured_proxy:
            continue
        if configured_proxy.startswith(("http:", "https:")):
            normalized_proxy = configured_proxy
        elif configured_proxy.startswith("//"):
            normalized_proxy = f"http:{configured_proxy}"
        else:
            normalized_proxy = f"http://{configured_proxy}"
        try:
            parsed_proxy = urlsplit(normalized_proxy)
            _ = parsed_proxy.port
        except ValueError as expected:
            if urllib3_parse_error or underlying.args == expected.args:
                return True
    return False


def _is_bedrock_local_configuration_value_error(exc: ValueError, *, session: object | None) -> bool:
    """Recognize ValueErrors proven to come from local AWS configuration."""
    integer_variables = list(_BEDROCK_INTEGER_ENVIRONMENT_VARIABLES)
    if os.environ.get("AWS_CSM_ENABLED", "").casefold() == "true":
        integer_variables.append("AWS_CSM_PORT")
    for variable in integer_variables:
        value = os.environ.get(variable)
        if value is None:
            continue
        try:
            int(value)
        except ValueError as expected:
            if exc.args == expected.args:
                return True

    botocore_session = getattr(session, "_session", None)
    get_config_variable = getattr(botocore_session, "get_config_variable", None)
    if callable(get_config_variable):
        config_variables = [
            "max_attempts",
            "metadata_service_num_attempts",
            "metadata_service_timeout",
        ]
        try:
            if get_config_variable("csm_enabled"):
                config_variables.append("csm_port")
        except Exception:
            pass
        for variable in config_variables:
            try:
                get_config_variable(variable)
            except ValueError as expected:
                if exc.args == expected.args:
                    return True
            except Exception:
                pass
        credential_process = _bedrock_credential_process(session)
        if credential_process is not None:
            try:
                shlex.split(credential_process)
            except ValueError as expected:
                if exc.args == expected.args:
                    return True

    if isinstance(exc, json.JSONDecodeError):
        get_service_model = getattr(botocore_session, "get_service_model", None)
        if callable(get_service_model):
            for service_name in (*_BEDROCK_RUNTIME_SERVICE_MODELS, *_bedrock_active_credential_service_models(session)):
                try:
                    get_service_model(service_name)
                except json.JSONDecodeError as expected:
                    if exc.args == expected.args:
                        return True
                except Exception:
                    pass

    for variable in _BEDROCK_ENDPOINT_ENVIRONMENT_VARIABLES:
        value = os.environ.get(variable)
        if value is None:
            continue
        if exc.args == (f"Invalid endpoint: {value}",):
            return True
        try:
            _ = urlsplit(value).port
        except ValueError as expected:
            if exc.args == expected.args:
                return True

    container_uri = os.environ.get("AWS_CONTAINER_CREDENTIALS_FULL_URI")
    if container_uri is not None:
        try:
            container_host = urlsplit(container_uri).hostname
        except ValueError:
            container_host = None
        if isinstance(container_host, str) and str(exc).startswith(f"Unsupported host '{container_host}'"):
            return True

    relative_uri = os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
    if relative_uri is not None and exc.args == (f"Invalid endpoint: http://169.254.170.2{relative_uri}",):
        return True

    if exc.args == ("Auth token value is not a legal header value",) and (
        "AWS_CONTAINER_AUTHORIZATION_TOKEN" in os.environ or "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE" in os.environ
    ):
        return True
    message = str(exc)
    return (
        message.startswith("Invalid endpoint: ")
        or message.startswith("Port could not be cast to integer value as ")
        or message.startswith("Unsupported host '")
        or message == "Auth token value is not a legal header value"
    )


def _is_native_catalog_endpoint(provider: ProviderConfig) -> bool:
    """Return whether catalog HTTP status has the provider's native meaning."""
    provider_name = provider.provider.casefold()
    if provider_name == "anthropic" and not provider.base_url:
        return True
    if not isinstance(provider.base_url, str) or provider.base_url != provider.base_url.strip():
        return False
    if "\\" in provider.base_url or any(character in provider.base_url for character in ("?", "#", ";")):
        return False
    try:
        endpoint = urlsplit(provider.base_url)
        port = endpoint.port
    except (TypeError, ValueError):
        return False
    if (
        endpoint.scheme.casefold() != "https"
        or endpoint.hostname is None
        or endpoint.username is not None
        or endpoint.password is not None
        or port not in {None, 443}
    ):
        return False

    if provider_name in {"openai", "openai-compatible"}:
        return endpoint.hostname.casefold() == "api.openai.com" and endpoint.path in {"/v1", "/v1/"}
    if provider_name == "nv_build":
        return endpoint.hostname.casefold() == "integrate.api.nvidia.com" and endpoint.path in {"/v1", "/v1/"}
    if provider_name == "anthropic":
        return endpoint.hostname.casefold() == "api.anthropic.com" and endpoint.path in {"", "/", "/v1", "/v1/"}
    return False


def _catalog_listing_is_authoritative(provider: ProviderConfig) -> bool:
    """Return whether absence from the listing proves the model is unusable."""
    # Anthropic aliases resolve through its single-model endpoint but are not
    # guaranteed to appear in the unique-ID listing. Bedrock's foundation-model
    # listing excludes other valid InvokeModel resources such as inference profiles.
    # OpenAI provisioned models can likewise be usable through Responses without
    # appearing in /models, so only NVIDIA Build's native listing is authoritative.
    return provider.provider.casefold() == "nv_build" and _is_native_catalog_endpoint(provider)


def credential_probe_disposition(
    provider: ProviderConfig,
    probe: ModelProbeResult,
) -> CredentialProbeDisposition:
    """Classify a live catalog probe without rejecting compatible custom gateways."""
    if probe.ok:
        provider_name = provider.provider.casefold()
        if provider_name == "bedrock":
            # Only native catalog + runtime endpoints prove AWS credentials.
            return (
                CredentialProbeDisposition.VERIFIED
                if getattr(probe, "catalog_authoritative", True)
                else CredentialProbeDisposition.DEGRADED
            )
        if provider_name == "nv_build" or not _is_native_catalog_endpoint(provider):
            # NVIDIA Build and compatible gateways may expose a public catalog,
            # so listing a model does not prove that inference credentials work.
            return CredentialProbeDisposition.DEGRADED
        return CredentialProbeDisposition.VERIFIED

    raw_kind = getattr(probe, "failure_kind", None)
    try:
        failure_kind = ModelCatalogFailureKind(raw_kind) if raw_kind is not None else None
    except (TypeError, ValueError):
        failure_kind = ModelCatalogFailureKind.UNKNOWN

    if failure_kind == ModelCatalogFailureKind.AUTHENTICATION:
        return (
            CredentialProbeDisposition.FATAL
            if provider.provider.casefold() == "bedrock" or _is_native_catalog_endpoint(provider)
            else CredentialProbeDisposition.DEGRADED
        )
    if failure_kind == ModelCatalogFailureKind.INVALID_CONFIGURATION:
        return CredentialProbeDisposition.FATAL
    if failure_kind == ModelCatalogFailureKind.MODEL_NOT_FOUND:
        return (
            CredentialProbeDisposition.FATAL
            if provider.provider.casefold() not in {"openai", "openai-compatible"}
            and _is_native_catalog_endpoint(provider)
            else CredentialProbeDisposition.DEGRADED
        )
    if failure_kind == ModelCatalogFailureKind.AUTHORIZATION:
        if provider.provider == "bedrock" or provider.provider.casefold() in {"openai", "openai-compatible"}:
            # ListFoundationModels permission is distinct from InvokeModel. OpenAI
            # restricted keys can likewise allow Responses while denying Models.
            return CredentialProbeDisposition.DEGRADED
        return (
            CredentialProbeDisposition.FATAL
            if _is_native_catalog_endpoint(provider)
            else CredentialProbeDisposition.DEGRADED
        )
    if failure_kind is None:
        # ``probe_model`` reached the catalog but did not find the selected model.
        return (
            CredentialProbeDisposition.FATAL
            if _catalog_listing_is_authoritative(provider)
            else CredentialProbeDisposition.DEGRADED
        )
    return CredentialProbeDisposition.DEGRADED


def _first_task_name(dataset: Path) -> str | None:
    for task_dir in sorted(path for path in dataset.iterdir() if path.is_dir() and not path.is_symlink()):
        if (task_dir / "task.toml").is_file():
            return task_dir.name
    return None


def _redact_detail(value: str, environment: Mapping[str, str]) -> str:
    secret_values = {item for item in environment.values() if len(item) >= 4}
    return redact_progress_detail(value, secret_values=secret_values)[-2000:]


def _first_trial_exception_detail(job_dir: Path) -> str:
    """Return a bounded first exception from Harbor's retained trial results."""
    try:
        result_paths = sorted(
            child / "result.json" for child in job_dir.iterdir() if child.is_dir() and (child / "result.json").is_file()
        )
    except OSError:
        return ""

    for result_path in result_paths:
        try:
            trial_result = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(trial_result, dict):
            continue
        candidates = [trial_result.get("exception_info")]
        step_results = trial_result.get("step_results")
        if isinstance(step_results, list):
            candidates.extend(
                step_result.get("exception_info") for step_result in step_results if isinstance(step_result, dict)
            )
        for exception_info in candidates:
            if not isinstance(exception_info, dict):
                continue
            exception_type = exception_info.get("exception_type")
            exception_message = exception_info.get("exception_message")
            parts = [
                part.strip() for part in (exception_type, exception_message) if isinstance(part, str) and part.strip()
            ]
            if parts:
                detail = " | ".join(" ".join(part.split()) for part in parts)
                return f"{result_path.parent.name}: {detail}"[:1500]
    return ""


def validate_harbor_agent_only_job_result(
    result_path: Path,
    *,
    expected_trials: int,
) -> tuple[bool, str]:
    """Validate a verification-disabled Harbor job and its agent result.

    Harbor 0.13.2 records an agent-only trial as completed at the job level,
    but intentionally leaves its evaluation trial and reward counts at zero.
    The per-trial result is therefore the proof that the agent actually ran.
    """
    if not isinstance(expected_trials, int) or isinstance(expected_trials, bool) or expected_trials <= 0:
        return False, f"Expected trial count is invalid: {expected_trials!r}"

    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False, f"Harbor exited successfully but did not produce {result_path}"
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"Harbor produced an unreadable agent-only job result at {result_path}: {exc}"

    if not isinstance(result, dict):
        return False, f"Harbor agent-only job result at {result_path} is not a JSON object"
    total = result.get("n_total_trials")
    stats = result.get("stats")
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0 or not isinstance(stats, dict):
        return False, f"Harbor agent-only job result at {result_path} is missing trial statistics"
    if total != expected_trials:
        return False, f"Harbor agent-only job declared {total} trials; expected {expected_trials}"

    counter_names = (
        "n_completed_trials",
        "n_errored_trials",
        "n_running_trials",
        "n_pending_trials",
        "n_cancelled_trials",
        "n_retries",
    )
    counters: dict[str, int] = {}
    for key in counter_names:
        value = stats.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return False, f"Harbor agent-only job result has invalid {key}: {value!r}"
        counters[key] = value

    for key, label in (
        ("n_errored_trials", "errored"),
        ("n_running_trials", "running"),
        ("n_pending_trials", "pending"),
        ("n_cancelled_trials", "cancelled"),
    ):
        if counters[key]:
            detail = f"Harbor agent-only job did not complete successfully: {counters[key]} {label}"
            if key == "n_errored_trials" and (exception_detail := _first_trial_exception_detail(result_path.parent)):
                detail = f"{detail}; first trial: {exception_detail}"
            return False, detail
    completed = counters["n_completed_trials"]
    if completed != total:
        return False, f"Harbor agent-only job did not complete successfully: completed {completed}/{total} trials"

    evals = stats.get("evals")
    if not isinstance(evals, dict) or not evals:
        return False, "Harbor agent-only job result has no evaluation statistics"
    for eval_name, eval_stats in evals.items():
        if not isinstance(eval_stats, dict):
            return False, f"Harbor agent-only evaluation {eval_name!r} has invalid statistics"
        for key in ("n_trials", "n_errors"):
            value = eval_stats.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value != 0:
                return False, f"Harbor agent-only evaluation {eval_name!r} has invalid {key}: {value!r}"
        if eval_stats.get("reward_stats") != {}:
            return False, f"Harbor agent-only evaluation {eval_name!r} has invalid reward_stats"

    job_dir = result_path.parent
    try:
        trial_result_paths = sorted(
            child / "result.json" for child in job_dir.iterdir() if child.is_dir() and (child / "result.json").is_file()
        )
    except OSError as exc:
        return False, f"Harbor agent-only trial results at {job_dir} are unreadable: {exc}"
    if len(trial_result_paths) != expected_trials:
        return False, (
            f"Harbor agent-only job did not produce {expected_trials} trial result(s); found {len(trial_result_paths)}"
        )

    for trial_result_path in trial_result_paths:
        try:
            trial_result = json.loads(trial_result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return False, f"Harbor produced an unreadable trial result at {trial_result_path}: {exc}"
        if not isinstance(trial_result, dict):
            return False, f"Harbor trial result at {trial_result_path} is not a JSON object"
        if "exception_info" not in trial_result:
            return False, f"Harbor trial result at {trial_result_path} is missing exception_info"
        if trial_result["exception_info"] is not None:
            return False, f"Harbor agent-only trial {trial_result_path.parent.name} recorded an exception"
        agent_result = trial_result.get("agent_result")
        step_results = trial_result.get("step_results")
        if isinstance(agent_result, dict):
            if step_results is not None:
                return False, (
                    f"Harbor agent-only trial {trial_result_path.parent.name} has mixed top-level and step agent results"
                )
            continue
        if agent_result is not None:
            return False, f"Harbor agent-only trial {trial_result_path.parent.name} has invalid agent_result"

        if step_results is None:
            return False, f"Harbor agent-only trial {trial_result_path.parent.name} has no agent result"
        if not isinstance(step_results, list):
            return False, f"Harbor agent-only trial {trial_result_path.parent.name} has invalid step_results"
        if not step_results:
            return False, f"Harbor agent-only trial {trial_result_path.parent.name} has no step results"
        for step_index, step_result in enumerate(step_results, start=1):
            if not isinstance(step_result, dict):
                return False, (
                    f"Harbor agent-only trial {trial_result_path.parent.name} has invalid step result {step_index}"
                )
            step_name = step_result.get("step_name")
            if not isinstance(step_name, str) or not step_name.strip():
                return False, (
                    f"Harbor agent-only trial {trial_result_path.parent.name} step {step_index} has invalid step_name"
                )
            if "exception_info" not in step_result:
                return False, (
                    f"Harbor agent-only trial {trial_result_path.parent.name} step {step_name!r} "
                    "is missing exception_info"
                )
            if step_result["exception_info"] is not None:
                return False, (
                    f"Harbor agent-only trial {trial_result_path.parent.name} step {step_name!r} recorded an exception"
                )
            if not isinstance(step_result.get("agent_result"), dict):
                return False, (
                    f"Harbor agent-only trial {trial_result_path.parent.name} step {step_name!r} has no agent result"
                )

    return True, ""


def _bedrock_client_error_kind(exc: ClientError) -> tuple[ModelCatalogFailureKind, int | None]:
    """Classify a sanitized AWS service error, including credential providers."""
    response = exc.response if isinstance(exc.response, dict) else {}
    error = response.get("Error") if isinstance(response.get("Error"), dict) else {}
    metadata = response.get("ResponseMetadata") if isinstance(response.get("ResponseMetadata"), dict) else {}
    error_code = str(error.get("Code") or "")
    http_status = metadata.get("HTTPStatusCode")
    if not isinstance(http_status, int) or isinstance(http_status, bool):
        http_status = None

    credential_bootstrap = getattr(exc, "operation_name", "") != "ListFoundationModels"
    if credential_bootstrap and (
        error_code in _BEDROCK_CREDENTIAL_BOOTSTRAP_TRANSIENT_CODES
        or http_status in {408, 429}
        or (isinstance(http_status, int) and http_status >= 500)
    ):
        return ModelCatalogFailureKind.UNAVAILABLE, http_status
    if credential_bootstrap:
        # A permanent upstream credential operation failure means InvokeModel
        # cannot be signed. It is distinct from catalog-only authorization.
        return ModelCatalogFailureKind.AUTHENTICATION, http_status
    if http_status == 401 or error_code in {
        "ExpiredTokenException",
        "ExpiredToken",
        "IncompleteSignature",
        "InvalidAccessKeyId",
        "InvalidClientTokenId",
        "InvalidSignatureException",
        "MissingAuthenticationToken",
        "SignatureDoesNotMatch",
        "UnrecognizedClientException",
    }:
        return ModelCatalogFailureKind.AUTHENTICATION, http_status
    if error_code == "RequestExpired":
        return ModelCatalogFailureKind.INVALID_CONFIGURATION, http_status
    if error_code in {"AccessDenied", "AccessDeniedException"}:
        return ModelCatalogFailureKind.AUTHORIZATION, http_status
    if error_code in {"ServiceUnavailable", "ServiceUnavailableException", "ThrottlingException"} or (
        isinstance(http_status, int) and (http_status == 429 or http_status >= 500)
    ):
        return ModelCatalogFailureKind.UNAVAILABLE, http_status
    return ModelCatalogFailureKind.UNKNOWN, http_status


def _bedrock_exception_result(
    provider: ProviderConfig,
    exc: Exception,
    *,
    catalog_request: bool,
    session: object | None,
) -> ModelProbeResult:
    """Convert a Bedrock SDK failure to a redacted, policy-safe result."""
    http_status = None
    if isinstance(exc, ClientError):
        failure_kind, http_status = _bedrock_client_error_kind(exc)
    elif (
        _is_bedrock_local_sdk_data_error(exc)
        or _is_bedrock_local_credential_file_error(exc)
        or _is_bedrock_local_credential_process_output_error(exc)
        or _is_bedrock_local_proxy_configuration_error(exc)
    ):
        failure_kind = ModelCatalogFailureKind.INVALID_CONFIGURATION
    elif isinstance(exc, _BEDROCK_AUTHENTICATION_ERRORS):
        failure_kind = ModelCatalogFailureKind.AUTHENTICATION
    elif isinstance(
        exc,
        (
            *_BEDROCK_INVALID_CONFIGURATION_ERRORS,
            FileNotFoundError,
            IsADirectoryError,
            NotADirectoryError,
            PermissionError,
        ),
    ) or (isinstance(exc, OSError) and exc.errno == errno.ENOEXEC and _bedrock_credential_process(session) is not None):
        failure_kind = ModelCatalogFailureKind.INVALID_CONFIGURATION
    elif (
        isinstance(exc, OSError)
        and exc.errno in _BEDROCK_CREDENTIAL_PROCESS_TRANSIENT_ERRNOS
        and _bedrock_credential_process(session) is not None
    ):
        failure_kind = ModelCatalogFailureKind.UNAVAILABLE
    elif _is_bedrock_local_cached_token_error(exc):
        failure_kind = ModelCatalogFailureKind.AUTHENTICATION
    elif isinstance(exc, ValueError):
        if _is_bedrock_local_configuration_value_error(exc, session=session):
            failure_kind = ModelCatalogFailureKind.INVALID_CONFIGURATION
        elif catalog_request:
            # Response parsers can raise ValueError for malformed timestamps or
            # other service/proxy data. That is not proof of bad local config.
            failure_kind = ModelCatalogFailureKind.INVALID_RESPONSE
        else:
            # Credential providers can likewise surface malformed remote data
            # while the client is being constructed. Preserve a degraded probe.
            failure_kind = ModelCatalogFailureKind.UNKNOWN
    elif isinstance(exc, BotocoreSSLError) and _is_invalid_bedrock_ca_bundle(session):
        failure_kind = ModelCatalogFailureKind.INVALID_CONFIGURATION
    elif isinstance(exc, BotoCoreError):
        failure_kind = ModelCatalogFailureKind.UNAVAILABLE
    else:
        failure_kind = ModelCatalogFailureKind.UNKNOWN
    return ModelProbeResult(
        False,
        provider.provider,
        provider.model,
        f"Bedrock model catalog request failed: {type(exc).__name__}",
        failure_kind=failure_kind,
        http_status=http_status,
    )


def _probe_bedrock_model(provider: ProviderConfig, *, timeout_seconds: float) -> ModelProbeResult:
    """Run one Bedrock catalog request inside the outer deadline worker."""
    session = None
    try:
        request_config = BotoConfig(
            connect_timeout=timeout_seconds,
            read_timeout=timeout_seconds,
            retries={"max_attempts": 0},
        )
        session = boto3.session.Session()
        _validate_bedrock_local_sdk_configuration(session)
        bearer_token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        if (
            isinstance(bearer_token, str)
            and bearer_token.strip()
            and any(not 0x21 <= ord(character) <= 0x7E for character in bearer_token)
        ):
            raise InvalidConfigError(error_msg="Bedrock bearer token contains invalid characters")
        if not isinstance(bearer_token, str) or not bearer_token.strip():
            metadata_unavailable = _monitor_bedrock_metadata_unavailability(session)
            get_credentials = getattr(session, "get_credentials", None)
            if callable(get_credentials):
                credentials = get_credentials()
                if credentials is None:
                    if metadata_unavailable[0]:
                        return ModelProbeResult(
                            False,
                            provider.provider,
                            provider.model,
                            "Bedrock credential metadata service was unavailable",
                            failure_kind=ModelCatalogFailureKind.UNAVAILABLE,
                        )
                    raise NoCredentialsError
                credentials.get_frozen_credentials()
        bedrock = session.client(
            "bedrock",
            region_name=provider.region or "us-west-2",
            config=request_config,
        )
        bedrock_runtime = session.client(
            "bedrock-runtime",
            region_name=provider.region or "us-west-2",
            config=request_config,
        )
        if (_bedrock_client_uses_tls(bedrock) or _bedrock_client_uses_tls(bedrock_runtime)) and (
            _is_invalid_bedrock_ca_bundle(session)
        ):
            raise InvalidConfigError(error_msg="configured AWS CA bundle cannot be loaded")
        region_name = provider.region or "us-west-2"
        catalog_authoritative = _is_native_bedrock_endpoint(
            session,
            bedrock,
            service_name="bedrock",
            region_name=region_name,
        ) and _is_native_bedrock_endpoint(
            session,
            bedrock_runtime,
            service_name="bedrock-runtime",
            region_name=region_name,
        )
    except Exception as exc:
        return _bedrock_exception_result(provider, exc, catalog_request=False, session=session)

    try:
        response = bedrock.list_foundation_models()
    except Exception as exc:
        return _bedrock_exception_result(provider, exc, catalog_request=True, session=session)
    summaries = response.get("modelSummaries") if isinstance(response, dict) else None
    available = {
        str(item["modelId"])
        for item in summaries or []
        if isinstance(item, dict) and isinstance(item.get("modelId"), str)
    }
    aliases = {provider.model}
    prefix, separator, unprefixed = provider.model.partition(".")
    if separator and prefix in {"apac", "eu", "global", "us"}:
        aliases.add(unprefixed)
    if aliases.isdisjoint(available):
        return ModelProbeResult(
            False,
            provider.provider,
            provider.model,
            f"model {provider.model} is not listed",
            catalog_authoritative=catalog_authoritative,
        )
    return ModelProbeResult(
        True,
        provider.provider,
        provider.model,
        f"model {provider.model} is available",
        catalog_authoritative=catalog_authoritative,
    )


def _probe_bedrock_model_with_deadline(
    provider: ProviderConfig,
    *,
    timeout_seconds: float,
) -> ModelProbeResult:
    """Bound session creation, credential providers, retries, and request I/O."""
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        return ModelProbeResult(
            False,
            provider.provider,
            provider.model,
            "model catalog timeout must be a positive number",
            failure_kind=ModelCatalogFailureKind.INVALID_CONFIGURATION,
        )

    result_queue: Queue[ModelProbeResult] = Queue(maxsize=1)
    deadline = monotonic() + timeout_seconds
    if not _BEDROCK_PROBE_SLOT.acquire(timeout=timeout_seconds):
        return ModelProbeResult(
            False,
            provider.provider,
            provider.model,
            "Bedrock model catalog request timed out",
            failure_kind=ModelCatalogFailureKind.UNAVAILABLE,
        )

    def run_probe() -> None:
        try:
            try:
                result = _probe_bedrock_model(provider, timeout_seconds=remaining)
            except Exception as exc:
                result = ModelProbeResult(
                    False,
                    provider.provider,
                    provider.model,
                    f"Bedrock model catalog request failed: {type(exc).__name__}",
                    failure_kind=ModelCatalogFailureKind.UNKNOWN,
                )
            result_queue.put_nowait(result)
        finally:
            _BEDROCK_PROBE_SLOT.release()

    remaining = deadline - monotonic()
    if remaining <= 0:
        _BEDROCK_PROBE_SLOT.release()
        return ModelProbeResult(
            False,
            provider.provider,
            provider.model,
            "Bedrock model catalog request timed out",
            failure_kind=ModelCatalogFailureKind.UNAVAILABLE,
        )

    try:
        worker = Thread(target=run_probe, name="bedrock-model-catalog-probe", daemon=True)
        worker.start()
    except BaseException as exc:
        _BEDROCK_PROBE_SLOT.release()
        if not isinstance(exc, Exception):
            raise
        return ModelProbeResult(
            False,
            provider.provider,
            provider.model,
            f"Bedrock model catalog request failed: {type(exc).__name__}",
            failure_kind=ModelCatalogFailureKind.UNKNOWN,
        )
    try:
        return result_queue.get(timeout=remaining)
    except Empty:
        return ModelProbeResult(
            False,
            provider.provider,
            provider.model,
            "Bedrock model catalog request timed out",
            failure_kind=ModelCatalogFailureKind.UNAVAILABLE,
        )


def probe_model(provider: ProviderConfig, *, timeout_seconds: float = 15.0) -> ModelProbeResult:
    """Check the selected model against the provider catalog within one deadline."""
    if provider.provider == "bedrock":
        return _probe_bedrock_model_with_deadline(provider, timeout_seconds=timeout_seconds)
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        return ModelProbeResult(
            False,
            provider.provider,
            provider.model,
            "model catalog timeout must be a positive number",
            failure_kind=ModelCatalogFailureKind.INVALID_CONFIGURATION,
        )

    deadline = monotonic() + timeout_seconds
    try:
        records = fetch_model_records(provider, timeout_seconds=timeout_seconds)
    except ModelCatalogError as exc:
        return ModelProbeResult(
            False,
            provider.provider,
            provider.model,
            str(exc),
            failure_kind=exc.kind,
            http_status=exc.http_status,
        )
    available = {record.id for record in records}
    if provider.model not in available:
        if provider.provider == "anthropic" and _is_native_catalog_endpoint(provider):
            remaining = deadline - monotonic()
            if remaining <= 0:
                return ModelProbeResult(
                    False,
                    provider.provider,
                    provider.model,
                    "model catalog request timed out",
                    failure_kind=ModelCatalogFailureKind.UNAVAILABLE,
                )
            try:
                resolved = fetch_anthropic_model_record(
                    provider,
                    provider.model,
                    timeout_seconds=remaining,
                )
            except ModelCatalogError as exc:
                if exc.http_status == 404:
                    return ModelProbeResult(
                        False,
                        provider.provider,
                        provider.model,
                        f"model {provider.model} is not available",
                        failure_kind=ModelCatalogFailureKind.MODEL_NOT_FOUND,
                        http_status=404,
                    )
                return ModelProbeResult(
                    False,
                    provider.provider,
                    provider.model,
                    str(exc),
                    failure_kind=exc.kind,
                    http_status=exc.http_status,
                )
            return ModelProbeResult(
                True,
                provider.provider,
                provider.model,
                f"model {provider.model} resolves to {resolved.id}",
            )
        return ModelProbeResult(False, provider.provider, provider.model, f"model {provider.model} is not listed")
    return ModelProbeResult(True, provider.provider, provider.model, f"model {provider.model} is available")


def run_agent_runtime_preflight(
    *,
    dataset: Path,
    agent: str,
    model: str,
    env_mode: str,
    jobs_dir: Path,
    run_env: Mapping[str, str],
    timeout_multiplier: float = 1.0,
    timeout_seconds: int = DEFAULT_PREFLIGHT_TIMEOUT_SECONDS,
    override_cpus: int | None = None,
    override_memory_mb: int | None = None,
    override_storage_mb: int | None = None,
    agent_import_path: str | None = None,
) -> PreflightResult:
    """Start one real agent task and stop before the full A/B matrix."""
    task_name = _first_task_name(dataset)
    job_name = f"runtime-preflight-{agent}"
    if task_name is None:
        return PreflightResult(False, agent, model, "No staged tasks are available for runtime preflight.", job_name)

    command = build_harbor_run_command(
        dataset_path=dataset,
        agent=agent,
        job_name=job_name,
        env_mode=env_mode,
        n_attempts=1,
        n_concurrent=1,
        model=model,
        jobs_dir=jobs_dir,
        timeout_multiplier=timeout_multiplier,
        disable_verification=True,
        include_task_names=[task_name],
        override_cpus=override_cpus,
        override_memory_mb=override_memory_mb,
        override_storage_mb=override_storage_mb,
        agent_import_path=agent_import_path,
    )
    try:
        handoff = _nvidia_build_key_handoff(run_env, env_mode=env_mode)
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            input=handoff.stdin_text,
            env=handoff.subprocess_env,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return PreflightResult(
            False,
            agent,
            model,
            f"Agent runtime preflight timed out after {timeout_seconds}s.",
            job_name,
        )
    except OSError as exc:
        return PreflightResult(False, agent, model, f"Agent runtime preflight could not start: {exc}", job_name)

    if completed.returncode != 0:
        output = "\n".join(part for part in (completed.stderr, completed.stdout) if part).strip()
        detail = _redact_detail(output, run_env) or f"harbor run exited {completed.returncode}"
        return PreflightResult(False, agent, model, detail, job_name)

    ok, detail = validate_harbor_agent_only_job_result(
        jobs_dir / job_name / "result.json",
        expected_trials=1,
    )
    return PreflightResult(ok, agent, model, _redact_detail(detail, run_env), job_name)
