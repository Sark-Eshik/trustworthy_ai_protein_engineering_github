# tests/infrastructure/test_config_loader.py
"""Unit tests for the centralized configuration loader framework."""

import os
import pytest
from src.infrastructure.config_loader import ConfigLoader, AppConfig


def test_config_loader_default_creation(tmp_path):
    """Test loader defaults compile cleanly when no configs are present."""
    loader = ConfigLoader(base_path=str(tmp_path))
    config = loader.load_config(env_override="development")

    assert isinstance(config, AppConfig)
    assert config.environment == "development"
    assert config.paths.configs_dir == "configs"
    assert config.paths.logs_dir == "logs"


def test_config_loader_yaml_merging(tmp_path):
    """Test environment merging and Pydantic validation handles structured files."""
    # Write synthetic paths config
    paths_file = tmp_path / "paths.yaml"
    with open(paths_file, "w", encoding="utf-8") as f:
        f.write(
            """
paths:
  configs_dir: "custom_configs"
  data_dir: "custom_data"
"""
        )

    # Write synthetic environment configuration
    dev_file = tmp_path / "development.yaml"
    with open(dev_file, "w", encoding="utf-8") as f:
        f.write(
            """
hardware:
  cpu_count: 8
  ram_limit_gb: 32.0
logging:
  level: "DEBUG"
"""
        )

    loader = ConfigLoader(base_path=str(tmp_path))
    config = loader.load_config(env_override="development")

    # Verify merging results
    assert config.environment == "development"
    assert config.paths.configs_dir == "custom_configs"
    assert config.paths.data_dir == "custom_data"
    assert config.hardware.cpu_count == 8
    assert config.hardware.ram_limit_gb == 32.0
    assert config.logging.level == "DEBUG"
