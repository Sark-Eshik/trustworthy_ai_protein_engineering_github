# docs/datasets.md
# Centralized Dataset Framework Documentation (Stage 2)

This document describes the design patterns, loading modules, validation procedures, and testing coverage implementing the Dataset Framework layer.

## Module Map
- `src/validation/schema_validator.py`: Crosschecks loaded DataFrames with registry definitions (columns names and types, primary keys uniqueness, null-value limitations) and serializes reports to `results/validation/schema_report.json`.
- `src/datasets/loaders.py`: Exposes a centralized loader class integrating with SchemaValidator to implement fail-fast file loading (Parquet, CSV, TSV) project-wide.
- `src/validation/dataset_validator.py`: Profiles dataset value ranges (dDG stability ranges, sequence bounds, SASA thresholds), logs missing values, writes profiling summaries to `reports/source_dataset_profile.md`, and certifies datasets in `reports/dataset_certification.md`.

## Active Core Source Datasets
1. **S001 - Megascale-D Mutation Dataset** (`data/raw/megascale_d/megascale_d.parquet`): Primary mutation benchmark containing dDG stabilities.
2. **S002 - Protein Sequence Dataset** (`data/raw/sequences/protein_sequences.parquet`): Input sequence catalog for ESM calculations.
3. **S003 - Protein Structure Dataset** (`data/raw/structures/protein_structures.parquet`): Structure inputs mapping to PDB coordinates and chains.

## Validation Gates
- Dataset schema matching rules.
- Primary key duplications constraints.
- Quantitative physical variables boundaries.
- No invalid files are written to disc or fed into calculations.

## Unit Testing Scope
- Validation on valid loaded files.
- Exception raising on invalid schema column sets.
- Out-of-bounds quantitative variable detections (dDG physical limits checks).
- File missing handlers checks.
