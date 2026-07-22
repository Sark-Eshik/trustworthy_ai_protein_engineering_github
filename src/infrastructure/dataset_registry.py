# src/infrastructure/dataset_registry.py
"""Centralized Dataset Registry module to define and catalog project datasets.

Defines schemas, paths, formats, and keys for all source, intermediate,
and output datasets used across the pipeline.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DatasetDefinition(BaseModel):
    """Metadata representing a single dataset definition in the registry."""

    dataset_id: str = Field(..., description="Unique dataset identifier (e.g., S001, D101)")
    name: str = Field(..., description="User-friendly name of the dataset")
    category: str = Field(..., description="Category of dataset: source, intermediate, final, or analysis")
    relative_path: str = Field(..., description="Standard path relative to workspace root")
    format: str = Field(..., description="File storage format (e.g., parquet, csv, fasta)")
    primary_key: str = Field(..., description="Primary identifier column in the dataset")
    required_columns: List[str] = Field(default_factory=list, description="List of columns that must be present")
    allowed_null_columns: List[str] = Field(default_factory=list, description="Columns permitted to have null values")


class DatasetRegistry:
    """Registry maintaining definitions for all datasets to ensure consistency."""

    def __init__(self):
        """Initialize the Dataset Registry and load the official specifications."""
        self._registry: Dict[str, DatasetDefinition] = {}
        self._load_official_definitions()

    def _load_official_definitions(self) -> None:
        """Populate official datasets from the project's data architecture specs."""
        definitions = [
            # Source Datasets
            DatasetDefinition(
                dataset_id="S001",
                name="Megascale-D Mutation Dataset",
                category="source",
                relative_path="data/raw/megascale_d/megascale_d.parquet",
                format="parquet",
                primary_key="mutation_id",
                required_columns=[
                    "mutation_id",
                    "protein_id",
                    "position",
                    "wildtype",
                    "mutant",
                    "experimental_ddg",
                ],
            ),
            DatasetDefinition(
                dataset_id="S002",
                name="Protein Sequence Dataset",
                category="source",
                relative_path="data/raw/sequences/protein_sequences.parquet",
                format="parquet",
                primary_key="protein_id",
                required_columns=["protein_id", "sequence", "sequence_length"],
            ),
            DatasetDefinition(
                dataset_id="S003",
                name="Protein Structure Dataset",
                category="source",
                relative_path="data/raw/structures/protein_structures.parquet",
                format="parquet",
                primary_key="protein_id",
                required_columns=["protein_id", "pdb_id", "chain_id", "structure_path"],
            ),
            # Sparsity Datasets
            DatasetDefinition(
                dataset_id="D101",
                name="Empirical Sparsity",
                category="intermediate",
                relative_path="data/intermediate/empirical/empirical_sparsity.parquet",
                format="parquet",
                primary_key="mutation_id",
                required_columns=[
                    "mutation_id",
                    "mutation_frequency",
                    "normalized_frequency",
                    "empirical_sparsity",
                ],
            ),
            DatasetDefinition(
                dataset_id="D102",
                name="Evolutionary Sparsity",
                category="intermediate",
                relative_path="data/intermediate/evolutionary/evolutionary_sparsity.parquet",
                format="parquet",
                primary_key="mutation_id",
                required_columns=[
                    "mutation_id",
                    "esm_probability",
                    "log_probability",
                    "evolutionary_sparsity",
                ],
            ),
            DatasetDefinition(
                dataset_id="D103",
                name="Structural Sparsity",
                category="intermediate",
                relative_path="data/intermediate/structural/structural_sparsity.parquet",
                format="parquet",
                primary_key="mutation_id",
                required_columns=["mutation_id", "sasa", "normalized_sasa", "structural_sparsity"],
            ),
            DatasetDefinition(
                dataset_id="D104",
                name="Combined Sparsity",
                category="intermediate",
                relative_path="data/intermediate/combined/combined_sparsity.parquet",
                format="parquet",
                primary_key="mutation_id",
                required_columns=[
                    "mutation_id",
                    "empirical_sparsity",
                    "evolutionary_sparsity",
                    "structural_sparsity",
                    "combined_sparsity",
                ],
                allowed_null_columns=["empirical_sparsity"],
            ),
            # Experiment 1
            DatasetDefinition(
                dataset_id="D201",
                name="Antisymmetry Predictions",
                category="analysis",
                relative_path="data/intermediate/experiment1/antisymmetry_predictions.parquet",
                format="parquet",
                primary_key="mutation_id",
                required_columns=["mutation_id", "forward_ddg", "reverse_ddg", "predictor"],
            ),
            DatasetDefinition(
                dataset_id="D202",
                name="Antisymmetry Results",
                category="analysis",
                relative_path="data/intermediate/experiment1/antisymmetry_results.parquet",
                format="parquet",
                primary_key="mutation_id",
                required_columns=[
                    "mutation_id",
                    "forward_ddg",
                    "reverse_ddg",
                    "antisymmetry_error",
                    "combined_sparsity",
                ],
            ),
            # Experiment 2
            DatasetDefinition(
                dataset_id="D301",
                name="Double Mutation Dataset",
                category="analysis",
                relative_path="data/raw/epistasis/double_mutants.parquet",
                format="parquet",
                primary_key="pair_id",
                required_columns=["pair_id", "mutation_a", "mutation_b", "experimental_ddg_ab"],
            ),
            DatasetDefinition(
                dataset_id="D302",
                name="Epistasis Results",
                category="analysis",
                relative_path="data/intermediate/experiment2/epistasis_results.parquet",
                format="parquet",
                primary_key="pair_id",
                required_columns=[
                    "pair_id",
                    "experimental_epistasis",
                    "predicted_epistasis",
                    "epistasis_error",
                    "combined_sparsity",
                ],
            ),
            # Integrated & Reliability
            DatasetDefinition(
                dataset_id="D401",
                name="Integrated Reliability Analysis",
                category="analysis",
                relative_path="data/final/reliability/integrated_reliability_analysis.parquet",
                format="parquet",
                primary_key="record_id",
                required_columns=[
                    "record_id",
                    "mutation_id",
                    "combined_sparsity",
                    "antisymmetry_error",
                    "epistasis_error",
                ],
            ),
            DatasetDefinition(
                dataset_id="D402",
                name="Reliability Scores",
                category="final",
                relative_path="data/final/reliability/reliability_scores.parquet",
                format="parquet",
                primary_key="mutation_id",
                required_columns=[
                    "mutation_id",
                    "combined_sparsity",
                    "reliability_score",
                    "reliability_category",
                ],
            ),
            # TKT Mutation Landscape
            DatasetDefinition(
                dataset_id="D501",
                name="TKT Mutation Landscape",
                category="final",
                relative_path="data/final/tkt/tkt_single_mutation_landscape.parquet",
                format="parquet",
                primary_key="mutation_id",
                required_columns=["mutation_id", "position", "wildtype", "mutant"],
            ),
            # TKT Mutation Analysis
            DatasetDefinition(
                dataset_id="D502",
                name="TKT Mutation Analysis",
                category="final",
                relative_path="data/final/tkt/tkt_mutation_analysis.parquet",
                format="parquet",
                primary_key="mutation_id",
                required_columns=[
                    "mutation_id",
                    "evolutionary_sparsity",
                    "structural_sparsity",
                    "combined_sparsity",
                    "reliability_score",
                    "predicted_stability",
                ],
            ),
            # Active-Site Analysis
            DatasetDefinition(
                dataset_id="D503",
                name="Active-Site Distance Analysis",
                category="final",
                relative_path="data/final/active_site/active_site_analysis.parquet",
                format="parquet",
                primary_key="mutation_id",
                required_columns=[
                    "mutation_id",
                    "distance_to_active_site",
                    "combined_sparsity",
                    "reliability_score",
                    "predicted_stability",
                ],
            ),
            # Industrial Fitness
            DatasetDefinition(
                dataset_id="D601",
                name="Industrial Fitness Scores",
                category="final",
                relative_path="data/final/industrial_fitness/industrial_fitness_scores.parquet",
                format="parquet",
                primary_key="mutation_id",
                required_columns=[
                    "mutation_id",
                    "reliability_score",
                    "predicted_stability",
                    "industrial_fitness_score",
                    "rank",
                ],
            ),
            # Top Candidate Mutations
            DatasetDefinition(
                dataset_id="D602",
                name="Top Candidate Mutations",
                category="final",
                relative_path="data/final/industrial_fitness/top_candidate_mutations.csv",
                format="csv",
                primary_key="mutation",
                required_columns=[
                    "rank",
                    "mutation",
                    "reliability_score",
                    "predicted_stability",
                    "industrial_fitness_score",
                ],
            ),
        ]

        for d in definitions:
            self._registry[d.dataset_id] = d

    def get_dataset(self, dataset_id: str) -> DatasetDefinition:
        """Retrieve the definition for a dataset ID.

        Parameters
        ----------
        dataset_id : str
            ID of the registered dataset (e.g., 'S001').

        Returns
        -------
        DatasetDefinition
            The schema and path definition.
        """
        if dataset_id not in self._registry:
            raise KeyError(f"Dataset with ID '{dataset_id}' is not registered in the system.")
        return self._registry[dataset_id]

    def list_datasets(self, category: Optional[str] = None) -> List[DatasetDefinition]:
        """List all datasets in the registry, optionally filtered by category.

        Parameters
        ----------
        category : Optional[str]
            Category name to filter by (e.g., 'source', 'intermediate').

        Returns
        -------
        List[DatasetDefinition]
            List of matching dataset definitions.
        """
        if category:
            return [d for d in self._registry.values() if d.category == category.lower()]
        return list(self._registry.values())


if __name__ == "__main__":
    # Exercise and Manual Validation entry point
    reg = DatasetRegistry()
    print("Initializing Dataset Registry...")

    # Load and check S001 (Megascale-D)
    s001_def = reg.get_dataset("S001")
    print(f"Dataset S001 loaded successfully.")

    # Validate output structure
    print("\n--- Manual Validation ---")
    print(f"Total datasets registered: {len(reg.list_datasets())}")
    print(f"S001 details:")
    print(f"  - Name: {s001_def.name}")
    print(f"  - Category: {s001_def.category}")
    print(f"  - Relative path: {s001_def.relative_path}")
    print(f"  - Primary Key: {s001_def.primary_key}")
    print(f"  - Required Columns: {s001_def.required_columns}")
    print("-------------------------")
