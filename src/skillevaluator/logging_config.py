# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logging configuration for SkillEvaluator."""

import logging

from rich.console import Console, ConsoleRenderable
from rich.logging import RichHandler

# Global console instance for rich output
console = Console()

# Log messages interpolate skill paths, provider errors, and tool output, so the
# handlers must not parse them as Rich markup: "[/x]" in a message would raise
# rich.errors.MarkupError out of the logging call itself, and "[link=...]"
# could spoof the log line.
_LOG_MARKUP = False


# While a live view owns the terminal, log records print through its console.
# Rich writes every other console past Live's stdout proxy, which leaves stale
# frame lines (a duplicated header) on screen.
_live_console: list[Console] = []


def route_logs_to(target: Console | None) -> None:
    """Print log records through *target* (a live view's console) until called with ``None``."""
    _live_console[:] = [target] if target is not None else []


class _LiteralRichHandler(RichHandler):
    """Rich handler that also drops terminal escape sequences from log messages."""

    def emit(self, record: logging.LogRecord) -> None:
        own = self.console
        self.console = _live_console[0] if _live_console else own
        try:
            super().emit(record)
        finally:
            self.console = own

    def render_message(self, record: logging.LogRecord, message: str) -> ConsoleRenderable:
        # Imported here: skillevaluator.utils imports this module.
        from skillevaluator.utils.rich_markup import strip_terminal_controls

        return super().render_message(record, strip_terminal_controls(message))


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
        handler = _LiteralRichHandler(
            console=console,
            show_time=False,
            show_path=False,
            markup=_LOG_MARKUP,
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
    handler = _LiteralRichHandler(
        console=console,
        show_time=False,
        show_path=False,
        markup=_LOG_MARKUP,
        rich_tracebacks=True,
    )
    handler.setLevel(level)

    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)

    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
