# src/infrastructure/config_loader.py
"""Centralized configuration loader with Pydantic validation.

Provides environment-based configuration merging paths, hardware, logging,
and dataset settings from YAML files.
"""

import os
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    """Configuration schema for directory and file paths."""

    configs_dir: str = Field(default="configs", description="Directory for configuration files")
    data_dir: str = Field(default="data", description="Directory for raw and processed datasets")
    docs_dir: str = Field(default="docs", description="Directory for documentation files")
    notebooks_dir: str = Field(default="notebooks", description="Directory for development notebooks")
    outputs_dir: str = Field(default="outputs", description="Directory for output files")
    reports_dir: str = Field(default="reports", description="Directory for generated reports")
    logs_dir: str = Field(default="logs", description="Directory for log files")
    checkpoints_dir: str = Field(default="checkpoints", description="Directory for application checkpoints")
    models_dir: str = Field(default="models", description="Directory for ML models and cached files")
    results_dir: str = Field(default="results", description="Directory for analysis results")
    tests_dir: str = Field(default="tests", description="Directory for test files")
    figures_dir: str = Field(default="figures", description="Directory for generated figures")
    papers_dir: str = Field(default="papers", description="Directory for publication paper materials")
    science_fair_dir: str = Field(default="science_fair", description="Directory for competition materials")
    presentations_dir: str = Field(default="presentations", description="Directory for presentations")
    release_dir: str = Field(default="release", description="Directory for final releases")


class HardwareConfig(BaseModel):
    """Configuration schema for hardware profiles and constraints."""

    cpu_count: Optional[int] = Field(default=None, description="Number of CPU cores to utilize")
    ram_limit_gb: Optional[float] = Field(default=None, description="RAM memory limit in Gigabytes")
    gpu_enabled: bool = Field(default=False, description="Whether to enable GPU processing if available")


class LoggingConfig(BaseModel):
    """Configuration schema for system logging framework."""

    level: str = Field(default="INFO", description="Global logging level (DEBUG, INFO, WARNING, ERROR)")
    log_to_file: bool = Field(default=True, description="Whether to write log outputs to a file")
    log_to_console: bool = Field(default=True, description="Whether to write log outputs to the console")


class ExperimentConfig(BaseModel):
    """Configuration schema for experiment tracking."""

    enabled: bool = Field(default=True, description="Whether experiment tracking is active")
    tracker_dir: str = Field(
        default="results/experiment_tracker",
        description="Path to store run metadata and experimental tracking files",
    )


class DatasetConfig(BaseModel):
    """Configuration schema for tracking dataset file paths."""

    megascale_d_path: str = Field(
        default="data/raw/megascale_d/megascale_d.parquet",
        description="Path to the primary Megascale-D mutation dataset",
    )
    protein_sequences_path: str = Field(
        default="data/raw/sequences/protein_sequences.parquet",
        description="Path to protein sequences dataset",
    )
    protein_structures_path: str = Field(
        default="data/raw/structures/protein_structures.parquet",
        description="Path to protein structures dataset",
    )


class AppConfig(BaseModel):
    """Unified application configuration schema containing all service configs."""

    environment: str = Field(default="development", description="Current operating mode (development, validation, production)")
    paths: PathsConfig = Field(default_factory=PathsConfig, description="Project-wide directories and paths config")
    hardware: HardwareConfig = Field(default_factory=HardwareConfig, description="Hardware constraints and allocation profile")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration details")
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig, description="Experiment tracker details")
    datasets: DatasetConfig = Field(default_factory=DatasetConfig, description="Registered paths for raw and validated datasets")


class ConfigLoader:
    """Loads and validates configuration files with merging capabilities.

    Retrieves configurations for paths, environment settings, and registries.
    """

    def __init__(self, base_path: str = "configs"):
        """Initialize the loader with the specified configuration directory.

        Parameters
        ----------
        base_path : str
            Directory path where configuration files are located.
        """
        self.base_path = base_path

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """Helper to read and parse a YAML file if it exists.

        Parameters
        ----------
        path : str
            Path to the YAML file.

        Returns
        -------
        Dict[str, Any]
            Parsed key-value pairs or empty dict if file does not exist.
        """
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
            return content if isinstance(content, dict) else {}

    def load_config(self, env_override: Optional[str] = None) -> AppConfig:
        """Loads and merges paths, global defaults, and environment-specific configs.

        Determines the operating environment, merges path overrides, merges environment
        overrides, and validates the output using Pydantic.

        Parameters
        ----------
        env_override : Optional[str]
            Operating environment override ("development", "validation", "production").
            If not provided, reads the 'SPARSITY_ENV' environment variable, defaulting to "development".

        Returns
        -------
        AppConfig
            Validated application configuration model.
        """
        # Determine environment
        environment = env_override or os.environ.get("SPARSITY_ENV", "development")
        environment = environment.lower()

        if environment not in ("development", "validation", "production"):
            environment = "development"

        # 1. Load paths.yaml if present
        paths_file = os.path.join(self.base_path, "paths.yaml")
        paths_data = self._load_yaml(paths_file)

        # 2. Load environment configuration (development.yaml, validation.yaml, or production.yaml)
        env_file = os.path.join(self.base_path, f"{environment}.yaml")
        env_data = self._load_yaml(env_file)

        # Build raw structure to feed into Pydantic AppConfig model
        raw_config: Dict[str, Any] = {
            "environment": environment,
            "paths": paths_data.get("paths", {}),
            "hardware": env_data.get("hardware", {}),
            "logging": env_data.get("logging", {}),
            "experiment": env_data.get("experiment", {}),
            "datasets": env_data.get("datasets", {}),
        }

        # Handle nested mergers if values are partially specified in env configs
        # Merge keys in paths if they were provided in environment-specific file too
        if "paths" in env_data and isinstance(env_data["paths"], dict):
            for k, v in env_data["paths"].items():
                raw_config["paths"][k] = v

        return AppConfig(**raw_config)


if __name__ == "__main__":
    # Exercise and Manual Validation entry point
    loader = ConfigLoader()
    try:
        config = loader.load_config()
        print("Configuration Loaded")
        print(f"{config.environment}.yaml detected")
        print("\n--- Manual Validation ---")
        print(f"Current mode: {config.environment}")
        print("Current paths:")
        for path_name, path_val in config.paths.model_dump().items():
            print(f"  - {path_name}: {path_val}")
        print("Current hardware profile:")
        print(f"  - CPU Count: {config.hardware.cpu_count}")
        print(f"  - RAM Limit (GB): {config.hardware.ram_limit_gb}")
        print(f"  - GPU Enabled: {config.hardware.gpu_enabled}")
        print("-------------------------")
    except Exception as e:
        print(f"FAILED: {e}")
