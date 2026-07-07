# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logging configuration for SkillEvaluator."""

import logging

from rich.console import Console
from rich.logging import RichHandler

# Global console instance for rich output
console = Console()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if no handlers exist
    if not logger.handlers:
        logger.setLevel(level)

        # Rich handler for console output
        handler = RichHandler(
            console=console,
            show_time=False,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        handler.setLevel(level)

        # Simple format - Rich handles the styling
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)

        logger.addHandler(handler)

        # Prevent duplicate logs when root logger also has handlers
        logger.propagate = False

    return logger


def setup_logging(verbose: bool = False) -> None:
    """Setup root logging configuration.

    Args:
        verbose: If True, set logging level to DEBUG
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add Rich handler
    handler = RichHandler(
        console=console,
        show_time=False,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    handler.setLevel(level)

    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)

    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
