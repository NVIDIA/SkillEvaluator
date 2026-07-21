# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry helpers for SkillEvaluator.

Telemetry is opt-in: configure a standard OTEL endpoint to export data.
"""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import hashlib
import logging
import os
import re
import time
import tomllib
from collections.abc import Iterable, Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

SERVICE_NAME = "skillevaluator"
SERVICE_VERSION_PACKAGE = "skillevaluator"
DEFAULT_DEPLOYMENT_ENV = "local"
DEFAULT_OTLP_PROTOCOL = "http/protobuf"
DEFAULT_OTLP_TIMEOUT_SECONDS = "2"

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_MAX_ATTR_LEN = int(os.environ.get("SKILLEVALUATOR_TELEMETRY_MAX_ATTR_LEN", "2048"))
_SECRET_KEY_PARTS = {
    "auth",
    "authorization",
    "bearer",
    "credential",
    "credentials",
    "key",
    "password",
    "private",
    "secret",
    "token",
}
_TOKEN_COUNT_KEYS = {
    "completion_tokens",
    "input_tokens",
    "n_input_tokens",
    "n_output_tokens",
    "output_tokens",
    "prompt_tokens",
    "last_token_usage",
    "token_count",
    "tokens",
    "total_tokens",
}
_SENSITIVE_KEY_PATTERN = (
    r"[a-z0-9_.-]*(?:api[_-]?key|secret|password|credential|authorization|bearer|token|"
    r"access[_-]?key|session[_-]?token|private[_-]?key)[a-z0-9_.-]*"
)
_AUTH_HEADER_RE = re.compile(r"(?im)\b(?P<key>(?:proxy-)?authorization)\s*:\s*(?P<scheme>[A-Za-z]+)\s+[^\r\n]+")
_SENSITIVE_QUOTED_ASSIGNMENT_RE = re.compile(
    rf"(?i)\b(?P<key>{_SENSITIVE_KEY_PATTERN})\s*(?P<sep>[:=])\s*(?P<quote>[\"'])(?P<value>[^\r\n]*?)(?P=quote)"
)
_SENSITIVE_COLON_ASSIGNMENT_RE = re.compile(rf"(?im)\b(?P<key>{_SENSITIVE_KEY_PATTERN})\s*(?P<sep>:)\s*[^\r\n,;]+")
_SENSITIVE_EQUALS_ASSIGNMENT_RE = re.compile(rf"(?i)\b(?P<key>{_SENSITIVE_KEY_PATTERN})\s*(?P<sep>=)\s*[^\s\"',;]+")
_REDACTIONS = (
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer <redacted>"),
    (re.compile(r"(?<![A-Za-z0-9_-])sk-[a-zA-Z0-9_-]{8,}"), "sk-<redacted>"),
    (re.compile(r"(?<![A-Za-z0-9_-])nvapi-[a-zA-Z0-9_-]{8,}"), "nvapi-<redacted>"),
    (re.compile(r"(?<![A-Za-z0-9_-])crsr_[a-f0-9]{16,}"), "crsr_<redacted>"),
    (re.compile(r"(?<![A-Za-z0-9_-])sha256~[A-Za-z0-9._~-]+"), "sha256~<redacted>"),
    (re.compile(r"(?<![A-Za-z0-9_-])(gh[opusr])_[A-Za-z0-9]{36,}"), r"\1_<redacted>"),
    (re.compile(r"(?<![A-Za-z0-9_-])AKIA[0-9A-Z]{16}(?![A-Za-z0-9])"), "AKIA<redacted>"),
    (
        re.compile(r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{16,}"),
        "<redacted-jwt>",
    ),
    (re.compile(r"(?<![A-Za-z0-9_-])(xox[baprs])-[0-9]{8,}-[A-Za-z0-9-]{8,}"), r"\1-<redacted>"),
    (re.compile(r"(?<![A-Za-z0-9_-])glpat-[A-Za-z0-9_-]{20,}"), "glpat-<redacted>"),
)
_HIGH_CARDINALITY_METRIC_KEYS = {
    "skillevaluator.run.id",
    "skillevaluator.eval.case_id",
    "skillevaluator.agent_eval.trial_name",
    "skillevaluator.results.dir",
    "skillevaluator.results.trial_dir",
    "skillevaluator.artifact.path",
    "skillevaluator.harbor.environment_dir",
    "skillevaluator.field_check.run_id",
    "trace_id",
    "span_id",
    "git.commit.sha",
}
_METRIC_ATTRIBUTE_KEYS = {
    "skillevaluator.agent.harness",
    "skillevaluator.agent.model",
    "skillevaluator.agent.model.normalized",
    "skillevaluator.agent.model_source",
    "skillevaluator.agent_eval.has_trajectory",
    "skillevaluator.agent_eval.lift.direction",
    "skillevaluator.agent_eval.runner",
    "skillevaluator.agent_eval.trajectory.readable",
    "skillevaluator.agent_eval.variant",
    "skillevaluator.command",
    "skillevaluator.dataset.dry_run",
    "skillevaluator.dataset.mode",
    "skillevaluator.dataset.no_llm",
    "skillevaluator.dataset.refined",
    "skillevaluator.dimension.name",
    "skillevaluator.env_mode",
    "skillevaluator.eval.tier",
    "skillevaluator.eval.static.enabled",
    "skillevaluator.eval.deep.enabled",
    "skillevaluator.agent_eval.enabled",
    "skillevaluator.finding.severity",
    "skillevaluator.finding.type",
    "skillevaluator.harbor.failure.kind",
    "skillevaluator.harbor.failure.reason",
    "skillevaluator.harbor.variant",
    "skillevaluator.execution.context",
    "skillevaluator.invocation.source",
    "skillevaluator.layer",
    "skillevaluator.metric.name",
    "skillevaluator.project.namespace",
    "skillevaluator.project.path",
    "skillevaluator.skill.grade",
    "skillevaluator.skill.has_errors",
    "skillevaluator.skill.has_warnings",
    "skillevaluator.skill.name",
    "skillevaluator.skill.pass",
    "skillevaluator.status",
    "skillevaluator.suggestion.category",
    "skillevaluator.suggestion.mode",
    "gen_ai.operation.name",
    "gen_ai.request.model",
    "gen_ai.system",
    "openinference.span.kind",
}
_METRIC_USER_ATTRIBUTE_KEYS = {"skillevaluator.user.hash", "skillevaluator.user.login"}
_RESOURCE_ENV_KEYS = ("deployment.environment.name", "deployment.environment", "env")
_CHILD_TELEMETRY_ENV_NAMES = {
    "SKILLEVALUATOR_TELEMETRY_ENABLED",
    "DD_ENV",
}

try:  # pragma: no cover - exercised in E2E smoke when dependencies are present.
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider
    from opentelemetry.sdk.metrics.export import AggregationTemporality, PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Status, StatusCode

    _OTEL_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on host env.
    metrics = trace = None  # type: ignore[assignment]
    OTLPMetricExporter = OTLPSpanExporter = None  # type: ignore[assignment]
    Counter = Histogram = MeterProvider = None  # type: ignore[assignment]
    AggregationTemporality = PeriodicExportingMetricReader = None  # type: ignore[assignment]
    Resource = TracerProvider = BatchSpanProcessor = None  # type: ignore[assignment]
    Status = StatusCode = None  # type: ignore[assignment]
    _OTEL_IMPORT_ERROR = exc

_initialized = False
_enabled = False
_tracer_provider: Any | None = None
_meter_provider: Any | None = None
_tracer: Any | None = None
_meter: Any | None = None
_instruments: dict[str, Any] = {}
_shutdown = False
_error_context: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar(
    "skillevaluator.telemetry.error_context",
    default=None,
)
_error_context_source_span_id: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "skillevaluator.telemetry.error_context_source_span_id",
    default=None,
)
_remember_error_context: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "skillevaluator.telemetry.remember_error_context",
    default=False,
)


def _env_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return None


def _telemetry_env_value(suffix: str, default: str | None = None) -> str | None:
    """Read SkillEvaluator telemetry configuration."""
    return os.environ.get(f"SKILLEVALUATOR_TELEMETRY_{suffix}", default)


def _telemetry_env_bool(suffix: str) -> bool | None:
    return _env_bool(f"SKILLEVALUATOR_TELEMETRY_{suffix}")


def _service_version() -> str:
    for parent in Path(__file__).resolve().parents:
        pyproject = parent / "pyproject.toml"
        if not pyproject.exists():
            continue
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            break
        project = data.get("project")
        if isinstance(project, Mapping) and isinstance(project.get("version"), str):
            return project["version"]
        break
    try:
        return version(SERVICE_VERSION_PACKAGE)
    except PackageNotFoundError:
        return "dev"


def _has_otlp_endpoint() -> bool:
    return any(
        os.environ.get(name)
        for name in (
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        )
    )


def _should_enable() -> bool:
    if _telemetry_env_bool("DISABLED") is True:
        return False
    return _telemetry_env_bool("ENABLED") is True


def _telemetry_explicitly_requested() -> bool:
    return _telemetry_env_bool("ENABLED") is True


def _endpoint(signal: str) -> str | None:
    specific = os.environ.get(f"OTEL_EXPORTER_OTLP_{signal.upper()}_ENDPOINT")
    if specific:
        return specific
    base = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not base:
        if _has_otlp_endpoint():
            logger.debug(
                "SkillEvaluator telemetry: no OTLP endpoint configured for signal %s; "
                "set OTEL_EXPORTER_OTLP_%s_ENDPOINT to enable it",
                signal,
                signal.upper(),
            )
            return None
        return None
    base = base.rstrip("/")
    if base.endswith(f"/v1/{signal}"):
        return base
    parsed = urlparse(base)
    if parsed.path.endswith("/v1/traces") or parsed.path.endswith("/v1/metrics"):
        base = base.rsplit("/v1/", 1)[0]
    return f"{base}/v1/{signal}"


def _metric_preferred_temporality() -> dict[type, Any] | None:
    if AggregationTemporality is None or Counter is None or Histogram is None:
        return None
    preference = os.environ.get("OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE", "delta").strip().lower()
    temporality = AggregationTemporality.CUMULATIVE if preference == "cumulative" else AggregationTemporality.DELTA
    return {
        Counter: temporality,
        Histogram: temporality,
    }


def _parse_headers(raw: str | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if not raw:
        return headers
    for item in raw.split(","):
        if not item.strip() or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            headers[key] = value
    return headers


def _parse_resource_attributes(raw: str | None) -> dict[str, str]:
    attrs: dict[str, str] = {}
    if not raw:
        return attrs
    for item in raw.split(","):
        if not item.strip() or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            attrs[key] = value
    return attrs


def _resource_env(attrs: Mapping[str, Any]) -> str | None:
    for key in _RESOURCE_ENV_KEYS:
        value = attrs.get(key)
        if value:
            return str(value)
    return None


def _configured_environment(resource_attrs: Mapping[str, Any] | None = None) -> str:
    attrs = resource_attrs or _parse_resource_attributes(os.environ.get("OTEL_RESOURCE_ATTRIBUTES"))
    return (
        os.environ.get("OTEL_DEPLOYMENT_ENVIRONMENT")
        or os.environ.get("DD_ENV")
        or os.environ.get("SKILLEVALUATOR_ENV")
        or _resource_env(attrs)
        or DEFAULT_DEPLOYMENT_ENV
    )


def deployment_environment(resource_attrs: Mapping[str, Any] | None = None) -> str:
    """Return the environment tag used for telemetry resources."""
    return _configured_environment(resource_attrs)


def _append_resource_attribute(raw: str | None, key: str, value: str) -> str:
    if not raw:
        return f"{key}={value}"
    attrs = _parse_resource_attributes(raw)
    if key in attrs:
        return raw
    return f"{raw.rstrip(',')},{key}={value}"


def _setdefault_env(name: str, value: str) -> None:
    if not os.environ.get(name):
        os.environ[name] = value


def _apply_default_environment() -> None:
    _setdefault_env("OTEL_SERVICE_NAME", SERVICE_NAME)
    _setdefault_env("OTEL_EXPORTER_OTLP_PROTOCOL", DEFAULT_OTLP_PROTOCOL)
    _setdefault_env("OTEL_EXPORTER_OTLP_TIMEOUT", DEFAULT_OTLP_TIMEOUT_SECONDS)
    resource_attrs = _parse_resource_attributes(os.environ.get("OTEL_RESOURCE_ATTRIBUTES"))
    env = _configured_environment(resource_attrs)
    _setdefault_env("DD_ENV", env)
    if not _resource_env(resource_attrs):
        os.environ["OTEL_RESOURCE_ATTRIBUTES"] = _append_resource_attribute(
            os.environ.get("OTEL_RESOURCE_ATTRIBUTES"),
            "deployment.environment.name",
            env,
        )


def child_process_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return env for subprocesses without propagating evaluator OTEL exporters.

    Docker, BuildKit, and some agent CLIs auto-enable OpenTelemetry when they
    see OTEL_* variables. SkillEvaluator records those child actions with
    parent spans, so child processes should not export their own low-level spans.
    """
    child_env = dict(os.environ if env is None else env)
    for key in list(child_env):
        if (
            key.startswith("OTEL_")
            or key.startswith("DD_")
            or key.startswith("SKILLEVALUATOR_TELEMETRY_")
            or key in _CHILD_TELEMETRY_ENV_NAMES
        ):
            child_env.pop(key, None)
    child_env["SKILLEVALUATOR_TELEMETRY_DISABLED"] = "true"
    return child_env


def _headers(signal: str) -> dict[str, str]:
    merged = _parse_headers(os.environ.get("OTEL_EXPORTER_OTLP_HEADERS"))
    merged.update(_parse_headers(os.environ.get(f"OTEL_EXPORTER_OTLP_{signal.upper()}_HEADERS")))
    return merged


def _resource_attributes(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "service.name": os.environ.get("OTEL_SERVICE_NAME", SERVICE_NAME),
        "service.version": os.environ.get("OTEL_SERVICE_VERSION", _service_version()),
        "telemetry.sdk.language": "python",
    }
    resource_attrs = _parse_resource_attributes(os.environ.get("OTEL_RESOURCE_ATTRIBUTES"))
    attrs.update(resource_attrs)
    env = _configured_environment(resource_attrs)
    if env:
        attrs["deployment.environment.name"] = env
        attrs["deployment.environment"] = env
        attrs["env"] = env
    for key in ("GITHUB_REPOSITORY", "GITHUB_RUN_ID", "GITHUB_JOB", "GITHUB_SHA"):
        value = os.environ.get(key)
        if value:
            attrs[key.lower().replace("_", ".")] = value
    if extra:
        attrs.update(flatten_attributes(extra))
    return sanitize_attributes(attrs)


def initialize_telemetry(resource_attributes: Mapping[str, Any] | None = None) -> bool:
    """Initialize OTel providers once and return whether export is active."""
    global _enabled, _initialized, _meter, _meter_provider, _tracer, _tracer_provider  # noqa: PLW0603 -- module-level OTel singletons initialized once
    if _initialized:
        return _enabled
    _initialized = True

    if not _should_enable():
        _enabled = False
        return False
    if _OTEL_IMPORT_ERROR is not None:
        message = "SkillEvaluator telemetry requested but OpenTelemetry is unavailable: %s"
        if _telemetry_explicitly_requested():
            logger.warning(message, _OTEL_IMPORT_ERROR)
        else:
            logger.debug(message, _OTEL_IMPORT_ERROR)
        _enabled = False
        return False

    traces_endpoint = _endpoint("traces")
    metrics_endpoint = _endpoint("metrics")
    if not traces_endpoint and not metrics_endpoint:
        _enabled = False
        return False

    resource = Resource.create(_resource_attributes(resource_attributes))
    try:
        if traces_endpoint:
            _tracer_provider = TracerProvider(resource=resource)
            _tracer_provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(
                        endpoint=traces_endpoint,
                        headers=_headers("traces") or None,
                        timeout=float(os.environ.get("OTEL_EXPORTER_OTLP_TIMEOUT", DEFAULT_OTLP_TIMEOUT_SECONDS)),
                    )
                )
            )
            trace.set_tracer_provider(_tracer_provider)
            _tracer = trace.get_tracer("skillevaluator.telemetry")

        if metrics_endpoint:
            reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(
                    endpoint=metrics_endpoint,
                    headers=_headers("metrics") or None,
                    timeout=float(os.environ.get("OTEL_EXPORTER_OTLP_TIMEOUT", DEFAULT_OTLP_TIMEOUT_SECONDS)),
                    preferred_temporality=_metric_preferred_temporality(),
                ),
                export_interval_millis=int(_telemetry_env_value("METRIC_EXPORT_MS", "1000") or "1000"),
            )
            _meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
            metrics.set_meter_provider(_meter_provider)
            _meter = metrics.get_meter("skillevaluator.telemetry")
    except Exception as exc:  # pragma: no cover - defensive around global provider state.
        logger.warning("Failed to initialize SkillEvaluator telemetry: %s", exc)
        _enabled = False
        return False

    _enabled = bool(_tracer or _meter)
    if _enabled:
        atexit.register(shutdown_telemetry)
    return _enabled


def is_telemetry_enabled() -> bool:
    return initialize_telemetry()


def flush_telemetry(timeout_millis: int = 5000) -> None:
    for provider in (_tracer_provider, _meter_provider):
        if provider is None:
            continue
        try:
            provider.force_flush(timeout_millis)
        except Exception:
            logger.debug("Telemetry force_flush failed", exc_info=True)


def shutdown_telemetry() -> None:
    global _shutdown  # noqa: PLW0603 -- module-level shutdown latch toggled once
    if _shutdown:
        return
    _shutdown = True
    for provider in (_tracer_provider, _meter_provider):
        if provider is None:
            continue
        try:
            provider.shutdown()
        except Exception:
            logger.debug("Telemetry shutdown failed", exc_info=True)


def _normalized_key_parts(key: str) -> tuple[str, set[str]]:
    camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(key or ""))
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", camel_split).strip("_").lower()
    parts = {part for part in normalized.split("_") if part}
    return normalized, parts


def _is_sensitive_key(key: str) -> bool:
    normalized, parts = _normalized_key_parts(key)
    if normalized in _TOKEN_COUNT_KEYS:
        return False
    compact = normalized.replace("_", "")
    if "api_key" in normalized or "apikey" in compact:
        return True
    if "accesskey" in compact or "privatekey" in compact or "sessiontoken" in compact:
        return True
    if "token" in parts or compact.endswith("token"):
        return True
    return bool(parts & _SECRET_KEY_PARTS)


def _redact_sensitive_assignment(match: re.Match[str]) -> str:
    key = match.group("key")
    if not _is_sensitive_key(key):
        return match.group(0)
    return f"{key}{match.group('sep')}<redacted>"


def _redact_auth_header(match: re.Match[str]) -> str:
    return f"{match.group('key')}: {match.group('scheme')} <redacted>"


def redact_sensitive_text(value: str, *, max_len: int | None = None) -> str:
    """Best-effort masking for credentials before telemetry or artifacts leave process memory."""
    out = value
    out = _AUTH_HEADER_RE.sub(_redact_auth_header, out)
    out = _SENSITIVE_QUOTED_ASSIGNMENT_RE.sub(_redact_sensitive_assignment, out)
    out = _SENSITIVE_COLON_ASSIGNMENT_RE.sub(_redact_sensitive_assignment, out)
    out = _SENSITIVE_EQUALS_ASSIGNMENT_RE.sub(_redact_sensitive_assignment, out)
    for pattern, replacement in _REDACTIONS:
        out = pattern.sub(replacement, out)
    if max_len is not None and len(out) > max_len:
        if max_len <= 14:
            return out[:max_len]
        return out[: max_len - 14] + "...<truncated>"
    return out


def _redact_string(value: str) -> str:
    return redact_sensitive_text(value, max_len=_MAX_ATTR_LEN)


def redact_sensitive_data(value: Any, *, parent_key: str = "", max_str_len: int | None = None) -> Any:
    """Recursively redact structured data using secret-looking key names."""
    if _is_sensitive_key(parent_key):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(key): redact_sensitive_data(item, parent_key=str(key), max_str_len=max_str_len)
            for key, item in value.items()
        }
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray, Mapping)):
        return [redact_sensitive_data(item, parent_key=parent_key, max_str_len=max_str_len) for item in value]
    if isinstance(value, str):
        return redact_sensitive_text(value, max_len=max_str_len)
    return value


def _clean_attr_value(key: str, value: Any) -> Any:
    if value is None:
        return None
    if _is_sensitive_key(key):
        return "<redacted>"
    if isinstance(value, Path):
        return _redact_string(str(value))
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, bool | int | float):
        return value
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        cleaned = []
        for item in value:
            if item is None:
                continue
            item_value = _clean_attr_value(key, item)
            if isinstance(item_value, str | bool | int | float):
                cleaned.append(item_value)
        return cleaned[:32]
    return _redact_string(str(value))


def flatten_attributes(attributes: Mapping[str, Any] | None, prefix: str = "") -> dict[str, Any]:
    if not attributes:
        return {}
    out: dict[str, Any] = {}
    for raw_key, value in attributes.items():
        key = f"{prefix}.{raw_key}" if prefix else str(raw_key)
        if isinstance(value, Mapping):
            out.update(flatten_attributes(value, key))
        else:
            out[key] = value
    return out


def sanitize_attributes(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in flatten_attributes(attributes).items():
        cleaned = _clean_attr_value(key, value)
        if cleaned is not None:
            sanitized[str(key)] = cleaned
    return sanitized


def user_identity_attributes() -> dict[str, Any]:
    mode = (_telemetry_env_value("IDENTITY_MODE", "team_only") or "team_only").strip().lower()
    raw = os.environ.get("GITHUB_ACTOR") or os.environ.get("USER") or os.environ.get("LOGNAME")
    attrs: dict[str, Any] = {}
    project = os.environ.get("GITHUB_REPOSITORY")
    namespace = os.environ.get("GITHUB_REPOSITORY_OWNER")
    if project:
        attrs["skillevaluator.project.path"] = project
    if namespace:
        attrs["skillevaluator.project.namespace"] = namespace
    if not raw or mode == "team_only":
        return attrs
    salt = _telemetry_env_value("HASH_SALT", "") or ""
    attrs["skillevaluator.user.hash"] = hashlib.sha256(f"{salt}:{raw}".encode()).hexdigest()[:16]
    if mode != "hashed":
        attrs["skillevaluator.user.login"] = raw
    return attrs


def _execution_context() -> str:
    if _env_bool("GITHUB_ACTIONS") is True or bool(os.environ.get("GITHUB_REPOSITORY")):
        return "ci"
    return "local_process"


def _invocation_source() -> str:
    return "direct"


def _invocation_attributes(*, include_trace_details: bool = True) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "skillevaluator.invocation.source": _invocation_source(),
        "skillevaluator.execution.context": _execution_context(),
    }
    if not include_trace_details:
        return attrs

    return attrs


def common_attributes(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    attrs = user_identity_attributes()
    attrs.update(
        {
            "git.branch": os.environ.get("GITHUB_REF_NAME") or os.environ.get("GIT_BRANCH") or "",
            "git.commit.sha": os.environ.get("GITHUB_SHA") or os.environ.get("GIT_COMMIT") or "",
            "ci.pipeline.id": os.environ.get("GITHUB_RUN_ID") or "",
            "ci.job.id": os.environ.get("GITHUB_JOB") or "",
        }
    )
    if extra:
        attrs.update(flatten_attributes(extra))
    attrs.update(_invocation_attributes())
    return sanitize_attributes({k: v for k, v in attrs.items() if v not in ("", None)})


def set_span_attributes(span: Any | None, attributes: Mapping[str, Any] | None) -> None:
    if span is None:
        return
    for key, value in sanitize_attributes(attributes).items():
        try:
            span.set_attribute(key, value)
        except Exception:
            logger.debug("Failed to set telemetry attribute %s", key, exc_info=True)


@contextlib.contextmanager
def trace_span(name: str, attributes: Mapping[str, Any] | None = None):
    initialize_telemetry()
    if _tracer is None:
        yield None
        return
    attrs = common_attributes(attributes)
    with _tracer.start_as_current_span(name, attributes=attrs) as span:
        try:
            yield span
        except BaseException as exc:
            remembered = _error_context.get() if _should_use_remembered_error(exc) else None
            if remembered:
                message, error_type = remembered
                if _error_context_source_span_id.get() != id(span):
                    mark_span_error(span, message, error_type=error_type)
            else:
                mark_span_error(span, exception=exc)
            raise


def _should_use_remembered_error(exc: BaseException) -> bool:
    if isinstance(exc, SystemExit):
        return True
    # Rich/status cleanup can raise EIO while unwinding an interrupt. Keep
    # the original child-span error as the root cause in telemetry.
    return isinstance(exc, OSError) and getattr(exc, "errno", None) == 5


def _recorded_exception_class(error_type: str) -> type[Exception]:
    class_name = re.sub(r"[^0-9A-Za-z_]", "_", error_type).strip("_")
    if not class_name or class_name[0].isdigit():
        class_name = "SkillEvaluatorTelemetryRecordedError"
    return type(class_name, (Exception,), {})


def mark_span_error(
    span: Any | None,
    message: str | None = None,
    *,
    exception: BaseException | None = None,
    error_type: str | None = None,
) -> None:
    """Attach error status and exception details to an active span."""
    if span is None:
        return
    raw_message = message
    if raw_message is None and exception is not None:
        raw_message = str(exception) or exception.__class__.__name__
    cleaned_message = _redact_string(str(raw_message or "unknown error"))
    resolved_type = error_type or (exception.__class__.__name__ if exception is not None else "SkillEvaluatorError")
    if _remember_error_context.get():
        _error_context.set((cleaned_message, resolved_type))
        _error_context_source_span_id.set(id(span))
    set_span_attributes(
        span,
        {
            "skillevaluator.error.message": cleaned_message,
            "skillevaluator.error.type": resolved_type,
            "error.message": cleaned_message,
            "error.type": resolved_type,
        },
    )
    try:
        try:
            raise _recorded_exception_class(resolved_type)(cleaned_message)
        except Exception as synthetic_exc:
            span.record_exception(synthetic_exc)
    except Exception:
        logger.debug("Failed to record telemetry span exception", exc_info=True)
    try:
        if Status is not None and StatusCode is not None:
            span.set_status(Status(StatusCode.ERROR, cleaned_message))
    except Exception:
        logger.debug("Failed to set telemetry span error status", exc_info=True)


@contextlib.contextmanager
def trace_command(command: str, layer: str, attributes: Mapping[str, Any] | None = None):
    attrs = {
        "skillevaluator.command": command,
        "skillevaluator.layer": layer,
        **(attributes or {}),
    }
    start = time.perf_counter()
    status = "ok"
    error_token = _error_context.set(None)
    error_source_token = _error_context_source_span_id.set(None)
    remember_token = _remember_error_context.set(True)
    try:
        with trace_span(f"skillevaluator.cli.{command}", attrs) as span:
            try:
                yield span
            except SystemExit as exc:
                status = "ok" if exc.code in (0, None) else "error"
                raise
            except BaseException:
                status = "error"
                raise
            finally:
                duration_ms = (time.perf_counter() - start) * 1000.0
                metric_attrs = {**attrs, "skillevaluator.status": status}
                set_span_attributes(span, {"skillevaluator.status": status, "skillevaluator.duration_ms": duration_ms})
                counter("skillevaluator.command.runs", 1, metric_attrs)
                histogram("skillevaluator.command.duration_ms", duration_ms, metric_attrs, unit="ms")
    finally:
        _remember_error_context.reset(remember_token)
        _error_context_source_span_id.reset(error_source_token)
        _error_context.reset(error_token)


def trace_evaluation(name: str, skill_name: str, *, tier: str | None = None, **attributes: Any):
    layer = "static" if tier == "tier1" or name == "tier1" else "deep" if tier == "tier2" or name == "tier2" else name
    span_name = (
        "skillevaluator.eval.static"
        if layer == "static"
        else "skillevaluator.eval.deep"
        if layer == "deep"
        else f"skillevaluator.eval.{name}"
    )
    attrs = {
        "skillevaluator.command": "evaluate",
        "skillevaluator.layer": layer,
        "skillevaluator.skill.name": skill_name,
        "skillevaluator.eval.tier": tier or name,
        **attributes,
    }
    return trace_span(span_name, attrs)


def _instrument(name: str, kind: str, unit: str = "1", description: str = "") -> Any | None:
    initialize_telemetry()
    if _meter is None:
        return None
    cache_key = f"{kind}:{name}"
    if cache_key in _instruments:
        return _instruments[cache_key]
    try:
        if kind == "counter":
            inst = _meter.create_counter(name, unit=unit, description=description)
        else:
            inst = _meter.create_histogram(name, unit=unit, description=description)
    except Exception:
        logger.debug("Failed to create telemetry instrument %s", name, exc_info=True)
        return None
    _instruments[cache_key] = inst
    return inst


def _metric_attributes(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    attrs = sanitize_attributes(attributes)
    include_user_tags = _telemetry_env_bool("METRIC_USER_TAGS") is True
    metric_attrs: dict[str, Any] = {}
    for key, value in attrs.items():
        if key in _HIGH_CARDINALITY_METRIC_KEYS:
            continue
        if key in _METRIC_USER_ATTRIBUTE_KEYS:
            if not include_user_tags:
                continue
        elif key not in _METRIC_ATTRIBUTE_KEYS:
            continue
        if isinstance(value, (bool, str)):
            metric_attrs[key] = value
    return metric_attrs


def _metric_attributes_with_invocation(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(attributes or {})
    merged.update(_invocation_attributes(include_trace_details=False))
    return _metric_attributes(merged)


def counter(name: str, value: int | float = 1, attributes: Mapping[str, Any] | None = None) -> None:
    inst = _instrument(name, "counter")
    if inst is None:
        return
    try:
        inst.add(value, _metric_attributes_with_invocation(attributes))
    except Exception:
        logger.debug("Failed to record telemetry counter %s", name, exc_info=True)


def histogram(
    name: str,
    value: int | float,
    attributes: Mapping[str, Any] | None = None,
    *,
    unit: str = "1",
) -> None:
    inst = _instrument(name, "histogram", unit=unit)
    if inst is None:
        return
    try:
        inst.record(float(value), _metric_attributes_with_invocation(attributes))
    except Exception:
        logger.debug("Failed to record telemetry histogram %s", name, exc_info=True)


def _record_agent_eval_lift(delta: float, attributes: Mapping[str, Any]) -> None:
    # OTel histograms accept only non-negative amounts. Preserve signed lift as
    # magnitude plus direction so negative regressions do not produce warnings.
    direction = "positive" if delta > 0 else "negative" if delta < 0 else "zero"
    histogram(
        "skillevaluator.agent_eval.lift",
        abs(delta),
        {**attributes, "skillevaluator.agent_eval.lift.direction": direction},
    )


def _record_agent_eval_dimension_lift(delta: float, attributes: Mapping[str, Any]) -> None:
    # Dimension lift has the same signed semantics as overall lift, but OTEL
    # histograms still require a non-negative recorded amount.
    direction = "positive" if delta > 0 else "negative" if delta < 0 else "zero"
    histogram(
        "skillevaluator.agent_eval.dimension_lift",
        abs(delta),
        {**attributes, "skillevaluator.agent_eval.lift.direction": direction},
    )


def record_tier1_evaluation(
    *,
    span: Any | None,
    correctness_score: float,
    discoverability_score: float,
    reliability_score: float,
    efficiency_score: float,
    overall_score: float,
    grade: str,
    issues_count: int,
    has_errors: bool,
    has_warnings: bool,
    metadata: Mapping[str, Any] | None = None,
    skill_name: str | None = None,
) -> None:
    attrs = {
        "skillevaluator.command": "evaluate",
        "skillevaluator.layer": "static",
        "skillevaluator.eval.tier": "tier1",
        "skillevaluator.skill.name": skill_name,
        "skillevaluator.skill.grade": grade,
        "skillevaluator.skill.issues_count": issues_count,
        "skillevaluator.skill.has_errors": has_errors,
        "skillevaluator.skill.has_warnings": has_warnings,
        "skillevaluator.skill.score.correctness": correctness_score,
        "skillevaluator.skill.score.discoverability": discoverability_score,
        "skillevaluator.skill.score.reliability": reliability_score,
        "skillevaluator.skill.score.efficiency": efficiency_score,
        "skillevaluator.skill.score.overall": overall_score,
        **(metadata or {}),
    }
    set_span_attributes(span, attrs)
    counter("skillevaluator.eval.static.runs", 1, attrs)
    for metric_name, value in {
        "correctness": correctness_score,
        "discoverability": discoverability_score,
        "reliability": reliability_score,
        "efficiency": efficiency_score,
        "overall": overall_score,
    }.items():
        histogram("skillevaluator.skill_quality.score", value, {**attrs, "skillevaluator.metric.name": metric_name})


def record_tier2_evaluation(
    *,
    span: Any | None,
    overall_score: float,
    overall_pass: bool,
    checks: Iterable[Any],
    summary: str,
    llm_calls: int,
    llm_tokens: int,
    skill_name: str | None = None,
) -> None:
    check_rows = list(checks)
    attrs = {
        "skillevaluator.command": "evaluate",
        "skillevaluator.layer": "deep",
        "skillevaluator.eval.tier": "tier2",
        "skillevaluator.skill.name": skill_name,
        "skillevaluator.skill.score.overall": overall_score,
        "skillevaluator.skill.pass": overall_pass,
        "skillevaluator.eval.check_count": len(check_rows),
        "gen_ai.usage.input_tokens": llm_tokens,
        "skillevaluator.llm.calls": llm_calls,
    }
    set_span_attributes(span, attrs)
    counter("skillevaluator.eval.deep.runs", 1, attrs)
    histogram("skillevaluator.skill_quality.score", overall_score, {**attrs, "skillevaluator.metric.name": "overall"})
    counter("skillevaluator.llm.calls", llm_calls, attrs)
    if llm_tokens:
        histogram("skillevaluator.llm.tokens", llm_tokens, attrs)


def record_dataset_creation(
    *,
    skill_name: str,
    cases_count: int,
    mode: str,
    refined: bool,
    no_llm: bool,
    dry_run: bool,
    duration_ms: float | None = None,
) -> None:
    attrs = {
        "skillevaluator.command": "create_dataset",
        "skillevaluator.layer": "dataset_creation",
        "skillevaluator.skill.name": skill_name,
        "skillevaluator.dataset.mode": mode,
        "skillevaluator.dataset.refined": refined,
        "skillevaluator.dataset.no_llm": no_llm,
        "skillevaluator.dataset.dry_run": dry_run,
    }
    counter("skillevaluator.dataset.creation.runs", 1, attrs)
    counter("skillevaluator.dataset.cases.generated", cases_count, attrs)
    if duration_ms is not None:
        histogram("skillevaluator.dataset.creation.duration_ms", duration_ms, attrs, unit="ms")


def _reward_overall_score(reward: Mapping[str, Any]) -> float | None:
    for key in ("overall", "score"):
        value = reward.get(key)
        if isinstance(value, int | float):
            return float(value)
    metrics = reward.get("metrics")
    if isinstance(metrics, Mapping):
        vals = [float(v) for v in metrics.values() if isinstance(v, int | float)]
        if vals:
            return sum(vals) / len(vals)
    return None


def _reward_metric_scores(reward: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for section_name in ("scores", "metrics", "custom_metrics"):
        section = reward.get(section_name)
        if isinstance(section, Mapping):
            for key, value in section.items():
                if isinstance(value, int | float):
                    out[str(key)] = float(value)
    details = reward.get("details")
    if isinstance(details, Mapping):
        for key, value in details.items():
            if isinstance(value, Mapping) and isinstance(value.get("score"), (int, float)):
                out[str(key)] = float(value["score"])
    return out


def _numeric_scores(value: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(value, Mapping):
        return out
    for key, score in value.items():
        if isinstance(score, int | float) and not isinstance(score, bool):
            out[str(key)] = float(score)
    return out


def _average_numeric_score(value: Any) -> float | None:
    scores = list(_numeric_scores(value).values())
    if not scores:
        return None
    return sum(scores) / len(scores)


def _lift_overall_score(agent_result: Mapping[str, Any], variant: str) -> float | None:
    lift = agent_result.get("lift")
    if not isinstance(lift, Mapping):
        return None
    overall = lift.get("overall")
    if not isinstance(overall, Mapping):
        return None
    value = overall.get(variant)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _dimension_scores(value: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(value, Mapping):
        return out
    for dimension, payload in value.items():
        if isinstance(payload, Mapping):
            score = payload.get("score")
            if isinstance(score, int | float) and not isinstance(score, bool):
                out[str(dimension)] = float(score)
    return out


def _bounded_label(value: Any, *, default: str = "unknown", max_len: int = 80) -> str:
    label = str(value or default).strip().lower()
    label = re.sub(r"[^a-z0-9_.-]+", "_", label).strip("._-")
    if not label:
        label = default
    return label[:max_len]


def record_harbor_scoring_gap(
    *,
    skill_name: str,
    agent: str,
    variant: str,
    expected_scored_attempts: int,
    actual_scored_attempts: int,
    env_mode: str | None = None,
    agent_model: str | None = None,
    agent_model_source: str | None = None,
    run_error: str | None = None,
    results_dir: str | Path | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> None:
    """Record a retained error signal when Harbor produced no scored rewards."""
    variant_label = _bounded_label(variant, default="unknown", max_len=40)
    attrs: dict[str, Any] = {
        "skillevaluator.command": "evaluate",
        "skillevaluator.layer": "agent_eval",
        "skillevaluator.agent_eval.runner": "harbor",
        "skillevaluator.agent.harness": agent,
        "skillevaluator.env_mode": env_mode,
        "skillevaluator.skill.name": skill_name,
        "skillevaluator.harbor.failure.kind": "no_scored_trials",
        "skillevaluator.harbor.failure.reason": (diagnostics or {}).get("failure_primary_reason"),
        "skillevaluator.harbor.variant": variant_label,
        "skillevaluator.harbor.expected_scored_attempts": expected_scored_attempts,
        "skillevaluator.harbor.actual_scored_attempts": actual_scored_attempts,
        "skillevaluator.status": "error",
        "skillevaluator.results.dir": str(results_dir) if results_dir else None,
        "openinference.span.kind": "EVALUATION",
    }
    if agent_model:
        attrs.update(
            _agent_model_attributes(
                agent,
                agent_models={agent: {"model": agent_model, "source": agent_model_source}},
            )
        )
    if diagnostics:
        attrs.update(flatten_attributes({"skillevaluator.harbor.diagnostics": diagnostics}))
    attrs = {k: v for k, v in attrs.items() if v not in ("", None)}

    message = f"{agent} {variant_label} produced {actual_scored_attempts}/{expected_scored_attempts} scored attempts"
    if run_error:
        message = f"{message}; {run_error}"
    with trace_span("skillevaluator.harbor.scoring_gap", attrs) as span:
        set_span_attributes(span, attrs)
        mark_span_error(span, message, error_type="SkillEvaluatorHarborNoScoredTrials")
    counter("skillevaluator.harbor.scoring_gaps", 1, attrs)


def _suggestion_category(value: Any) -> str:
    text = str(value or "").lower()
    if any(
        token in text
        for token in ("dockerfile", "container", "environment", "install", "missing cli", "command not found")
    ):
        return "environment"
    if any(token in text for token in ("unsafe", "secret", "destructive", "security", "permission")):
        return "safety"
    if any(token in text for token in ("skill.md", "discover", "workflow", "documentation", "instructions")):
        return "skill_docs"
    if any(token in text for token in ("evals.json", "expected_behavior", "ground_truth", "test cases", "dataset")):
        return "dataset"
    if any(token in text for token in ("additional agents", "claude-code", "codex", "openhands", "cross-agent")):
        return "agent_coverage"
    return "general"


_MODEL_PREFIX_ALIASES = {
    "anthropic",
    "aws",
    "azure",
    "bedrock",
    "google",
    "nvidia",
    "openai",
}


def normalize_agent_model(model: Any) -> str | None:
    """Return a stable model label for Datadog grouping without mutating raw model data."""
    text = str(model or "").strip()
    if not text or text.lower() in {"n/a", "none", "null", "unknown"}:
        return None

    text = text.replace(":", "/")
    text = re.sub(r"/+", "/", text).strip("/")
    if not text:
        return None

    parts = [part for part in text.split("/") if part]
    collapsed: list[str] = []
    for part in parts:
        normalized_part = part.lower()
        if collapsed and normalized_part in _MODEL_PREFIX_ALIASES and collapsed[-1].lower() == normalized_part:
            continue
        collapsed.append(part)
    return "/".join(collapsed) or None


def _agent_model_attributes(
    agent: str,
    *,
    agent_models: Mapping[str, Any] | None = None,
    results: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    candidates: list[Any] = []
    if agent_models:
        candidates.append(agent_models.get(agent))
    if results:
        agents_data = results.get("agents")
        if isinstance(agents_data, Mapping):
            agent_data = agents_data.get(agent)
            if isinstance(agent_data, Mapping):
                candidates.append(agent_data.get("model_resolution"))
                if agent_data.get("model"):
                    candidates.append(
                        {
                            "model": agent_data.get("model"),
                            "source": agent_data.get("model_source"),
                        }
                    )
        run_config = results.get("run_config")
        if isinstance(run_config, Mapping):
            run_config_agents = run_config.get("agents")
            if isinstance(run_config_agents, Mapping):
                candidates.append(run_config_agents.get(agent))

    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            attrs = {"skillevaluator.agent.model": candidate, "gen_ai.request.model": candidate}
            normalized = normalize_agent_model(candidate)
            if normalized:
                attrs["skillevaluator.agent.model.normalized"] = normalized
            return attrs
        if not isinstance(candidate, Mapping):
            continue
        model = candidate.get("model")
        if not model:
            continue
        model_text = str(model)
        attrs = {
            "skillevaluator.agent.model": model_text,
            "gen_ai.request.model": model_text,
        }
        normalized = normalize_agent_model(model_text)
        if normalized:
            attrs["skillevaluator.agent.model.normalized"] = normalized
        source = candidate.get("source") or candidate.get("model_source")
        if source:
            attrs["skillevaluator.agent.model_source"] = str(source)
        return attrs
    return {}


def record_agent_eval_findings(
    *,
    skill_name: str,
    agent: str,
    runner: str,
    findings: Iterable[Mapping[str, Any]],
    suggestions: Iterable[Any] | None = None,
    env_mode: str | None = None,
    agent_model: str | None = None,
    agent_model_source: str | None = None,
    suggestion_mode: str | None = None,
    artifact_path: str | Path | None = None,
) -> None:
    finding_rows = [dict(row) for row in findings]
    suggestion_rows = list(suggestions or [])
    severity_counts: dict[str, int] = {}
    for row in finding_rows:
        severity = _bounded_label(row.get("severity"))
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    base_attrs = {
        "skillevaluator.command": "evaluate",
        "skillevaluator.layer": "agent_eval",
        "skillevaluator.agent_eval.runner": runner,
        "skillevaluator.agent.harness": agent,
        "skillevaluator.env_mode": env_mode,
        "skillevaluator.skill.name": skill_name,
        "skillevaluator.findings.count": len(finding_rows),
        "skillevaluator.findings.critical_count": severity_counts.get("critical", 0),
        "skillevaluator.findings.warning_count": severity_counts.get("warning", 0),
        "skillevaluator.findings.ok_count": severity_counts.get("ok", 0),
        "skillevaluator.suggestions.count": len(suggestion_rows),
        "skillevaluator.artifact.path": str(artifact_path) if artifact_path else None,
        "openinference.span.kind": "EVALUATION",
    }
    if agent_model:
        base_attrs.update(
            _agent_model_attributes(
                agent,
                agent_models={agent: {"model": agent_model, "source": agent_model_source}},
            )
        )

    with trace_span("skillevaluator.agent_eval.findings", base_attrs):
        finding_groups: dict[tuple[str, str, str], int] = {}
        for row in finding_rows:
            metric_name = _bounded_label(row.get("metric"), default="overall")
            severity = _bounded_label(row.get("severity"))
            finding_type = _bounded_label(row.get("type"), default="metric_summary")
            finding_groups[(metric_name, severity, finding_type)] = (
                finding_groups.get((metric_name, severity, finding_type), 0) + 1
            )
            attrs = {
                **base_attrs,
                "skillevaluator.metric.name": metric_name,
                "skillevaluator.finding.severity": severity,
                "skillevaluator.finding.type": finding_type,
            }
            score = row.get("score")
            if isinstance(score, int | float):
                histogram("skillevaluator.agent_eval.finding.score", float(score), attrs)

        for (metric_name, severity, finding_type), count in finding_groups.items():
            counter(
                "skillevaluator.agent_eval.findings.count",
                count,
                {
                    **base_attrs,
                    "skillevaluator.metric.name": metric_name,
                    "skillevaluator.finding.severity": severity,
                    "skillevaluator.finding.type": finding_type,
                },
            )

        suggestion_groups: dict[tuple[str, str], int] = {}
        for suggestion in suggestion_rows:
            category = _suggestion_category(suggestion)
            mode = _bounded_label(suggestion_mode, default="remediation")
            suggestion_groups[(category, mode)] = suggestion_groups.get((category, mode), 0) + 1

        for (category, mode), count in suggestion_groups.items():
            counter(
                "skillevaluator.agent_eval.suggestions.count",
                count,
                {
                    **base_attrs,
                    "skillevaluator.suggestion.category": category,
                    "skillevaluator.suggestion.mode": mode,
                },
            )


def record_agent_trial(
    *,
    skill_name: str,
    agent: str,
    runner: str,
    variant: str,
    reward: Mapping[str, Any],
    trial_dir: str | Path | None = None,
    env_mode: str | None = None,
    agent_model: str | None = None,
    agent_model_source: str | None = None,
) -> None:
    trial_name = str(reward.get("_trial_name") or reward.get("trial_name") or "unknown")
    case_id = str(reward.get("entry_id") or trial_name.split("__", 1)[0] or "unknown")
    score = _reward_overall_score(reward)
    attrs = {
        "skillevaluator.command": "evaluate",
        "skillevaluator.layer": "agent_eval",
        "skillevaluator.agent_eval.runner": runner,
        "skillevaluator.agent_eval.variant": variant,
        "skillevaluator.agent.harness": agent,
        "skillevaluator.env_mode": env_mode,
        "skillevaluator.skill.name": skill_name,
        "skillevaluator.eval.case_id": case_id,
        "skillevaluator.agent_eval.trial_name": trial_name,
        "skillevaluator.agent_eval.has_trajectory": bool(reward.get("_has_trajectory") or reward.get("has_trajectory")),
        "skillevaluator.results.trial_dir": str(trial_dir) if trial_dir else None,
        "gen_ai.operation.name": "execute_agent",
        "gen_ai.system": agent,
        "openinference.span.kind": "AGENT",
    }
    attrs.update(
        _agent_model_attributes(
            agent,
            agent_models={
                agent: {
                    "model": agent_model,
                    "source": agent_model_source,
                }
            }
            if agent_model
            else None,
        )
    )
    trajectory_summary = reward.get("_trajectory_summary")
    if isinstance(trajectory_summary, Mapping):
        attrs.update(
            {
                "skillevaluator.agent_eval.trajectory.readable": trajectory_summary.get("readable"),
                "skillevaluator.agent_eval.trajectory.steps": trajectory_summary.get("steps"),
                "skillevaluator.agent_eval.trajectory.tool_calls": trajectory_summary.get("tool_calls"),
                "skillevaluator.agent_eval.trajectory.unique_tools": trajectory_summary.get("unique_tools"),
                "skillevaluator.agent_eval.trajectory.tool_names": trajectory_summary.get("tool_names"),
            }
        )
    if score is not None:
        attrs["skillevaluator.agent_eval.score.overall"] = score
    with trace_span("skillevaluator.agent_eval.trial", attrs) as span:
        set_span_attributes(span, attrs)
        counter("skillevaluator.agent_eval.trials.completed", 1, attrs)
        if score is not None:
            histogram("skillevaluator.agent_eval.score", score, {**attrs, "skillevaluator.metric.name": "overall"})
        for metric_name, metric_score in _reward_metric_scores(reward).items():
            histogram(
                "skillevaluator.agent_eval.score", metric_score, {**attrs, "skillevaluator.metric.name": metric_name}
            )


def record_agent_eval_summary(
    *,
    runner: str,
    skill_name: str,
    agents: Iterable[str],
    env_mode: str | None = None,
    results: Mapping[str, Any] | None = None,
    agent_models: Mapping[str, Any] | None = None,
    planned_trials: int | None = None,
    planned_containers: int | None = None,
    planned_pods: int | None = None,
    duration_ms: float | None = None,
) -> None:
    agent_list = [str(agent) for agent in agents]
    base_attrs = {
        "skillevaluator.command": "evaluate",
        "skillevaluator.layer": "agent_eval",
        "skillevaluator.agent_eval.runner": runner,
        "skillevaluator.env_mode": env_mode,
        "skillevaluator.skill.name": skill_name,
    }
    for agent_name in agent_list:
        attrs = {
            **base_attrs,
            "skillevaluator.agent.harness": agent_name,
            **_agent_model_attributes(agent_name, agent_models=agent_models, results=results),
        }
        counter("skillevaluator.agent_eval.runs", 1, attrs)
        if planned_trials:
            counter("skillevaluator.agent_eval.trials.planned", planned_trials, attrs)
        if planned_containers:
            counter("skillevaluator.harbor.containers.planned", planned_containers, attrs)
        if planned_pods:
            counter("skillevaluator.harbor.environments.planned", planned_pods, attrs)
        if duration_ms is not None:
            histogram("skillevaluator.agent_eval.duration_ms", duration_ms, attrs, unit="ms")
    if not results:
        return
    agents_data = results.get("agents") if isinstance(results, Mapping) else None
    if isinstance(agents_data, Mapping):
        for agent, data in agents_data.items():
            if not isinstance(data, Mapping):
                continue
            model_attrs = _agent_model_attributes(str(agent), agent_models=agent_models, results=results)
            completed_trials = sum(
                int(data.get(key) or 0)
                for key in ("num_trials_with", "num_trials_without")
                if isinstance(data.get(key), int | float)
            )
            if completed_trials:
                result_attrs = {
                    **base_attrs,
                    "skillevaluator.agent.harness": str(agent),
                    **model_attrs,
                }
                counter("skillevaluator.agent_eval.results.completed", completed_trials, result_attrs)
                if env_mode == "docker":
                    counter(
                        "skillevaluator.harbor.containers.started",
                        completed_trials,
                        result_attrs,
                    )
            for variant_key, scores in (
                ("with_skill", data.get("with_skill")),
                ("without_skill", data.get("without_skill")),
            ):
                if not isinstance(scores, Mapping):
                    continue
                emitted_metric_names: set[str] = set()
                for metric_name, value in scores.items():
                    if isinstance(value, int | float):
                        emitted_metric_names.add(str(metric_name))
                        histogram(
                            "skillevaluator.agent_eval.score",
                            float(value),
                            {
                                **base_attrs,
                                "skillevaluator.agent.harness": str(agent),
                                "skillevaluator.agent_eval.variant": variant_key,
                                "skillevaluator.metric.name": str(metric_name),
                                **model_attrs,
                            },
                        )
                overall_score = _lift_overall_score(data, variant_key)
                if overall_score is None:
                    overall_score = _average_numeric_score(scores)
                if overall_score is not None and "overall" not in emitted_metric_names:
                    histogram(
                        "skillevaluator.agent_eval.score",
                        overall_score,
                        {
                            **base_attrs,
                            "skillevaluator.agent.harness": str(agent),
                            "skillevaluator.agent_eval.variant": variant_key,
                            "skillevaluator.metric.name": "overall",
                            **model_attrs,
                        },
                    )
            for variant_key, dimensions_key in (
                ("with_skill", "dimensions_with_skill"),
                ("without_skill", "dimensions_without_skill"),
            ):
                for dimension_name, score in _dimension_scores(data.get(dimensions_key)).items():
                    histogram(
                        "skillevaluator.agent_eval.dimension_score",
                        score,
                        {
                            **base_attrs,
                            "skillevaluator.agent.harness": str(agent),
                            "skillevaluator.agent_eval.variant": variant_key,
                            "skillevaluator.dimension.name": dimension_name,
                            **model_attrs,
                        },
                    )
            with_dimensions = _dimension_scores(data.get("dimensions_with_skill"))
            without_dimensions = _dimension_scores(data.get("dimensions_without_skill"))
            for dimension_name in sorted(set(with_dimensions) & set(without_dimensions)):
                _record_agent_eval_dimension_lift(
                    with_dimensions[dimension_name] - without_dimensions[dimension_name],
                    {
                        **base_attrs,
                        "skillevaluator.agent.harness": str(agent),
                        "skillevaluator.dimension.name": dimension_name,
                        **model_attrs,
                    },
                )
            lift = data.get("lift")
            if isinstance(lift, Mapping):
                for metric_name, row in lift.items():
                    if isinstance(row, Mapping) and isinstance(row.get("delta"), (int, float)):
                        _record_agent_eval_lift(
                            float(row["delta"]),
                            {
                                **base_attrs,
                                "skillevaluator.agent.harness": str(agent),
                                "skillevaluator.metric.name": str(metric_name),
                                **model_attrs,
                            },
                        )


def _nested_average_scores(value: Any, *, score_key: str = "average_score") -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(value, Mapping):
        return out
    for metric_name, payload in value.items():
        if isinstance(payload, int | float):
            out[str(metric_name)] = float(payload)
        elif isinstance(payload, Mapping) and isinstance(payload.get(score_key), (int, float)):
            out[str(metric_name)] = float(payload[score_key])
    return out
