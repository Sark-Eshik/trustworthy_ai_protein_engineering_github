# src/infrastructure/logger.py
"""Centralized logging framework for console and file-based logging.

Ensures proper directories are created, formatting is consistent, and log levels
are applied project-wide.
"""

import logging
import os
from typing import Optional


def get_logger(
    name: str,
    log_dir: str = "logs",
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """Create and configure a robust logger with console and file handlers.

    Ensures the log directory is automatically created if missing, and configures
    consistent output formats.

    Parameters
    ----------
    name : str
        Name of the logger (usually __name__ of the calling module).
    log_dir : str
        Directory where log files are stored. Defaults to 'logs'.
    level : str
        Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_to_file : bool
        Whether to enable file-based logging.
    log_to_console : bool
        Whether to enable console-based logging.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    # Map string representation of level to logging integer constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Prevent duplicating handlers if get_logger is called multiple times on the same logger
    if logger.handlers:
        return logger

    # Console Handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_format = logging.Formatter(
            fmt="[%(levelname)s] %(name)s: %(message)s"
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    # File Handler
    if log_to_file:
        try:
            os.makedirs(log_dir, exist_ok=True)
            file_path = os.path.join(log_dir, "system.log")
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_format = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
        except OSError as e:
            # Revert to print if log directory cannot be created
            print(f"CRITICAL: Failed to create or access log directory '{log_dir}': {e}")

    return logger


if __name__ == "__main__":
    # Exercise and Manual Validation entry point
    test_logger = get_logger("system", log_dir="logs", level="INFO")
    test_logger.info("Initializing system diagnostics...")
    test_logger.warning("Diagnostics warning check: standard operational parameters active.")
    test_logger.info("Logging framework verified successfully.")

    # Validate output file presence
    log_file_path = os.path.join("logs", "system.log")
    if os.path.exists(log_file_path):
        print("\n--- Manual Validation ---")
        print(f"Log file successfully created at: {log_file_path}")
        print("Recent contents:")
        with open(log_file_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[-3:]:
                print(f"  {line.strip()}")
        print("-------------------------")
    else:
        print("FAILED: Log file was not created.")
