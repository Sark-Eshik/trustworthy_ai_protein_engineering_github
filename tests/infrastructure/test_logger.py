# tests/infrastructure/test_logger.py
"""Unit tests for the centralized logging framework."""

import os
import logging
import pytest
from src.infrastructure.logger import get_logger


def test_logger_creation(tmp_path):
    """Test logger initialization and file write verification."""
    log_dir = str(tmp_path / "logs")
    logger = get_logger(name="test_logger", log_dir=log_dir, level="INFO")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"

    logger.info("Test record message.")

    # Confirm system.log gets created
    log_file = os.path.join(log_dir, "system.log")
    assert os.path.exists(log_file)

    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Test record message." in content


def test_logger_duplicate_handlers():
    """Test logger handler deduction on sequential requests to prevent flooding."""
    logger_first = get_logger(name="duplicate_test", level="INFO")
    first_handler_count = len(logger_first.handlers)

    logger_second = get_logger(name="duplicate_test", level="INFO")
    second_handler_count = len(logger_second.handlers)

    assert first_handler_count == second_handler_count
