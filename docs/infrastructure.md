# docs/infrastructure.md
# Centralized Infrastructure Foundation Documentation

This document describes the software architectures, libraries, and design conventions implementing the Trustworthy AI Protein Engineering Foundation (Phase 1).

## Complete Package Map
- `src/infrastructure/config_loader.py`: Merges multi-stage environments (`development`, `validation`, `production`), merges common directory scopes (`paths.yaml`), and validates configurations strictly using Pydantic.
- `src/infrastructure/logger.py`: Implements process-safe, file-based (`logs/system.log`), and stdout-based diagnostic logging with strict string level mapping.
- `src/infrastructure/experiment_tracker.py`: Records individual run metrics, git context revisions, and global execution list files (`run_index.jsonl`).
- `src/infrastructure/checkpoint_manager.py`: Serializes process structures to dynamically restore pipeline states on interruption.
- `src/infrastructure/dataset_registry.py`: Registers official file formats, paths, and core columns for all 15 source, sparsity, and final analysis datasets.
- `src/infrastructure/validation_engine.py`: Encapsulates schema matching, primary key uniqueness constraints, null-value checks, and numeric range enforcement.
- `src/infrastructure/hardware_detection.py`: Profiles system CPU, virtual memory availability, and GPU acceleration environments (CUDA or Metal Apple Silicon unified configurations) dynamically.
- `src/bootstrap.py`: Validates complete directory structure trees (16 scopes), profiles hardware, loads registries, and certifies repository state.

## Schema Validation Criteria
Every intermediate data file must pass:
1. Column conformities (all registered variables present).
2. Primary key uniqueness checks (0 duplicate records).
3. Null limits enforcement (0 Null values in columns not specified in registry as allowed-null fields).
4. Boundary checks (quantitative values must reside within valid range definitions).

## Testing Scopes
- Config Merging & Overrides.
- Logger formatting and console output duplication prevention.
- Checkpoint serialization and restore.
- Dataset definitions lookup.
- Schema conformity and validation ranges.
