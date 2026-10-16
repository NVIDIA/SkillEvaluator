# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubeconfig path resolution and multi-file merging for Harbor GKE mode."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_KUBECONFIG_FILE_PATH_KEYS = frozenset(
    {
        "certificate-authority",
        "client-certificate",
        "client-key",
        "tokenFile",
        "idp-certificate-authority",
    }
)

_MERGED_KUBECONFIG_CACHE: dict[str, Path] = {}


def _resolve_relative_kubeconfig_path(
    raw_value: str,
    base_dir: Path | None,
    fallback_dirs: Sequence[Path] = (),
) -> str:
    """Resolve a relative kubeconfig path against its source file directory."""
    expanded = Path(raw_value.strip()).expanduser()
    if expanded.is_absolute():
        return str(expanded)
    if base_dir is not None:
        return str((base_dir / expanded).resolve())
    for directory in fallback_dirs:
        candidate = (directory / expanded).resolve()
        if candidate.exists():
            return str(candidate)
    if fallback_dirs:
        return str((fallback_dirs[0] / expanded).resolve())
    return raw_value


def _unwrap_kubeconfig_node(
    obj: Any,
    source_path: str | None = None,
    fallback_dirs: Sequence[Path] = (),
) -> Any:
    """Recursively unwrap Kubernetes ConfigNode wrappers while absolutizing relative file paths."""
    node_path = getattr(obj, "path", None) or source_path
    if hasattr(obj, "value"):
        return _unwrap_kubeconfig_node(obj.value, source_path=node_path, fallback_dirs=fallback_dirs)
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        base_dir = Path(node_path).expanduser().resolve().parent if node_path else None
        for k, v in obj.items():
            unwrapped = _unwrap_kubeconfig_node(v, source_path=node_path, fallback_dirs=fallback_dirs)
            if (
                isinstance(unwrapped, str)
                and unwrapped.strip()
                and (k in _KUBECONFIG_FILE_PATH_KEYS or (k == "command" and ("/" in unwrapped or os.sep in unwrapped)))
            ):
                unwrapped = _resolve_relative_kubeconfig_path(unwrapped, base_dir, fallback_dirs)
            out[k] = unwrapped
        return out
    if isinstance(obj, list):
        return [_unwrap_kubeconfig_node(item, source_path=node_path, fallback_dirs=fallback_dirs) for item in obj]
    return obj


def _resolve_single_kubeconfig(raw_kubeconfig: str | None = None) -> Path | None:
    """Resolve a single existing kubeconfig file path from KUBECONFIG or default location.

    If multiple path entries are specified, merge them into a single temporary kubeconfig
    file preserving all clusters, contexts, and credentials, restricted to permissions 0600.
    """
    raw = raw_kubeconfig if raw_kubeconfig is not None else os.environ.get("KUBECONFIG")
    if raw:
        valid_candidates: list[Path] = []
        for entry in raw.split(os.pathsep):
            cleaned = entry.strip()
            if cleaned:
                candidate = Path(cleaned).expanduser().resolve()
                if candidate.is_file():
                    valid_candidates.append(candidate)
        if valid_candidates:
            if len(valid_candidates) == 1:
                return valid_candidates[0]
            paths_str = os.pathsep.join(str(p) for p in valid_candidates)
            cache_key = os.pathsep.join(f"{p}:{p.stat().st_mtime_ns}:{p.stat().st_size}" for p in valid_candidates)
            cached_path = _MERGED_KUBECONFIG_CACHE.get(cache_key)
            if cached_path is not None and cached_path.is_file():
                return cached_path
            try:
                import atexit
                import tempfile

                import yaml
                from kubernetes.config.kube_config import KubeConfigMerger

                with tempfile.TemporaryDirectory(prefix="kubeconfig_stage_") as stage_dir:
                    staged_paths: list[str] = []
                    for idx, cand in enumerate(valid_candidates):
                        cand_data = yaml.safe_load(cand.read_text(encoding="utf-8"))
                        if isinstance(cand_data, dict):
                            cand_data = _unwrap_kubeconfig_node(cand_data, source_path=str(cand))
                            staged_file = Path(stage_dir) / f"cand_{idx}.yaml"
                            staged_file.write_text(
                                yaml.safe_dump(cand_data, default_flow_style=False),
                                encoding="utf-8",
                            )
                            staged_file.chmod(0o600)
                            staged_paths.append(str(staged_file))
                        else:
                            staged_paths.append(str(cand))
                    merger = KubeConfigMerger(os.pathsep.join(staged_paths))
                    fallback_dirs = [p.parent for p in valid_candidates]
                    merged_dict = _unwrap_kubeconfig_node(merger.config, fallback_dirs=fallback_dirs)
                if not merged_dict or not isinstance(merged_dict, dict):
                    logger.error("KubeConfigMerger produced empty or invalid configuration from %s", paths_str)
                    return None
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    prefix="merged_kubeconfig_",
                    suffix=".yaml",
                    delete=False,
                ) as handle:
                    merged_path = Path(handle.name)
                    yaml.safe_dump(merged_dict, handle, default_flow_style=False)
                    merged_path.chmod(0o600)
                _MERGED_KUBECONFIG_CACHE[cache_key] = merged_path
                atexit.register(lambda: merged_path.unlink(missing_ok=True))
                logger.info(
                    "Merged %d kubeconfig files into temporary file %s",
                    len(valid_candidates),
                    merged_path,
                )
                return merged_path
            except Exception as exc:
                logger.error(
                    "Failed to merge multiple kubeconfigs (%s): %s",
                    raw,
                    exc,
                )
                return None
        return None
    default_path = (Path.home() / ".kube" / "config").resolve()
    return default_path if default_path.is_file() else None
