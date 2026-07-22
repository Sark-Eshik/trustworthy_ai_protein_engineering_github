# src/infrastructure/validation_engine.py
"""Centralized Validation Engine for schema and data range validation.

Enforces schema conformity, missing value tolerances, primary key uniqueness,
and range constraints on datasets throughout the pipeline.
"""

import os
import json
import pandas as pd
from typing import Any, Dict, List, Optional
from src.infrastructure.dataset_registry import DatasetDefinition, DatasetRegistry


class ValidationReport(pd.DataFrame):
    """Validation report context wrapper containing structural check lists."""

    pass


class ValidationEngine:
    """Validator to enforce primary keys, schemas, nulls, and range parameters."""

    def __init__(self, registry: Optional[DatasetRegistry] = None):
        """Initialize the validation engine with an optional custom dataset registry.

        Parameters
        ----------
        registry : Optional[DatasetRegistry]
            Dataset registry containing official dataset definitions. Defaults to a new instance.
        """
        self.registry = registry or DatasetRegistry()

    def validate_schema(self, dataset_id: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Verify schema compliance, primary key uniqueness, and required columns.

        Parameters
        ----------
        dataset_id : str
            Registered identifier of the dataset (e.g., 'S001').
        df : pd.DataFrame
            DataFrame loaded from storage to validate.

        Returns
        -------
        Dict[str, Any]
            Status dictionary containing 'valid' boolean and descriptive comments.
        """
        definition = self.registry.get_dataset(dataset_id)
        errors: List[str] = []

        # 1. Enforce Required Columns
        for col in definition.required_columns:
            if col not in df.columns:
                errors.append(f"Missing required column: '{col}'")

        # 2. Enforce Primary Key Uniqueness
        pk = definition.primary_key
        if pk in df.columns:
            if df[pk].duplicated().any():
                num_duplicates = df[pk].duplicated().sum()
                errors.append(f"Primary key '{pk}' contains {num_duplicates} duplicate(s).")
            if df[pk].isnull().any():
                num_null_pk = df[pk].isnull().sum()
                errors.append(f"Primary key '{pk}' contains {num_null_pk} null/missing value(s).")
        else:
            errors.append(f"Primary key '{pk}' column missing from dataset.")

        # 3. Missing values validation (columns not in allowed_null_columns should not be null)
        for col in df.columns:
            if col in definition.required_columns and col not in definition.allowed_null_columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    errors.append(f"Column '{col}' is not allowed to contain nulls but has {null_count} null(s).")

        is_valid = len(errors) == 0
        return {
            "dataset_id": dataset_id,
            "name": definition.name,
            "valid": is_valid,
            "errors": errors,
        }

    def validate_ranges(
        self,
        dataset_id: str,
        df: pd.DataFrame,
        range_rules: Dict[str, tuple[float, float]],
    ) -> Dict[str, Any]:
        """Enforce range validation bounds on quantitative columns.

        Parameters
        ----------
        dataset_id : str
            Registered identifier of the dataset.
        df : pd.DataFrame
            DataFrame containing columns to constrain.
        range_rules : Dict[str, tuple[float, float]]
            Mapping of column names to inclusive (min_val, max_val) bounds.

        Returns
        -------
        Dict[str, Any]
            Status dictionary containing 'valid' boolean and range boundary failures.
        """
        errors: List[str] = []

        for col, (min_val, max_val) in range_rules.items():
            if col in df.columns:
                # Exclude nulls safely during numeric comparisons
                col_data = df[col].dropna()
                out_of_bounds = col_data[(col_data < min_val) | (col_data > max_val)]
                if not out_of_bounds.empty:
                    num_violations = len(out_of_bounds)
                    min_observed = out_of_bounds.min()
                    max_observed = out_of_bounds.max()
                    errors.append(
                        f"Column '{col}' range violation: {num_violations} value(s) "
                        f"outside [{min_val}, {max_val}] bounds. "
                        f"Observed range: [{min_observed}, {max_observed}]."
                    )
            else:
                errors.append(f"Range check failed: configured column '{col}' is missing.")

        is_valid = len(errors) == 0
        return {
            "dataset_id": dataset_id,
            "valid": is_valid,
            "errors": errors,
        }


if __name__ == "__main__":
    # Exercise and Manual Validation entry point
    engine = ValidationEngine()
    print("Testing Validation Engine...")

    # Create dummy DataFrame matching S001 schema
    valid_data = pd.DataFrame(
        {
            "mutation_id": ["mut1", "mut2", "mut3"],
            "protein_id": ["protA", "protA", "protB"],
            "position": [12, 14, 15],
            "wildtype": ["A", "G", "C"],
            "mutant": ["V", "C", "T"],
            "experimental_ddg": [1.2, -0.4, 3.4],
        }
    )

    invalid_data = pd.DataFrame(
        {
            "mutation_id": ["mut1", "mut1", None],  # Duplicate and Null PK
            "protein_id": ["protA", "protA", "protB"],
            "position": [12, 14, 15],
            "wildtype": ["A", "G", "C"],
            "mutant": ["V", "C", "T"],
            # missing experimental_ddg
        }
    )

    valid_report = engine.validate_schema("S001", valid_data)
    invalid_report = engine.validate_schema("S001", invalid_data)

    print("\n--- Manual Validation ---")
    print("Valid Dataset Report:")
    print(f"  - Valid: {valid_report['valid']}")
    print(f"  - Errors found: {len(valid_report['errors'])}")

    print("\nInvalid Dataset Report (Simulated Failures):")
    print(f"  - Valid: {invalid_report['valid']}")
    print(f"  - Errors found: {len(invalid_report['errors'])}")
    for err in invalid_report["errors"]:
        print(f"    * {err}")

    # Range check simulation (e.g. dDG values between -10 and +10)
    range_check = engine.validate_ranges("S001", valid_data, {"experimental_ddg": (-5.0, 5.0)})
    print(f"\nRange check (Valid Data [-5.0, 5.0]): Valid={range_check['valid']}")

    bad_range_check = engine.validate_ranges("S001", valid_data, {"experimental_ddg": (-0.1, 1.0)})
    print(f"Range check (Violations Expected): Valid={bad_range_check['valid']}")
    for err in bad_range_check["errors"]:
        print(f"    * {err}")
    print("-------------------------")
