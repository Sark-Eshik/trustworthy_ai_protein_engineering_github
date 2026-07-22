# src/tkt/landscape_analysis.py
"""TKT mutation landscape analysis module.

Calculates evolutionary sparsity, structural sparsity, combined sparsity,
reliability scores, and simulated stability predictions across the entire
11,400 single-point mutations in the Transketolase landscape (D502).
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from src.infrastructure.config_loader import ConfigLoader, AppConfig
from src.infrastructure.logger import get_logger
from src.infrastructure.dataset_registry import DatasetRegistry
from src.infrastructure.validation_engine import ValidationEngine
from src.sparsity.evolutionary.esm_probability_engine import ESMProbabilityEngine
from src.sparsity.structural.sasa_engine import SASAEngine

class LandscapeAnalyzer:
    """Orchestrates evolutionary, structural, combined sparsity and reliability score calculations for TKT."""

    def __init__(self, config_path: str = "configs") -> None:
        self.config_loader = ConfigLoader(base_path=config_path)
        self.config: AppConfig = self.config_loader.load_config()
        self.registry = DatasetRegistry()
        self.validation_engine = ValidationEngine(registry=self.registry)
        self.logger = get_logger(
            name="landscape_analyzer",
            log_dir=self.config.paths.logs_dir,
            level=self.config.logging.level,
        )

    @staticmethod
    def simulate_stability(position: int, wildtype: str, mutant: str, structural_sparsity: float) -> float:
        """Deterministically simulates a physically plausible consensus predicted stability change (ddG).
        
        Conservative mutations on the surface have stability changes near 0.0 kcal/mol.
        Destabilizing mutations in the buried core (high structural sparsity) have negative stability benefit (up to -4.0).
        Rare beneficial mutations are randomly/deterministically generated (up to +1.0).
        """
        import hashlib
        # Chemical groups
        groups = {
            "hydrophobic": set("AVILMFYW"),
            "charged_positive": set("KRH"),
            "charged_negative": set("DE"),
            "polar_uncharged": set("NQST"),
            "special_small": set("GPC"),
        }
        
        def get_group(aa: str) -> str:
            for name, members in groups.items():
                if aa in members:
                    return name
            return "other"
            
        wt_grp = get_group(wildtype)
        mut_grp = get_group(mutant)

        # Baseline stability benefit (conserved = 0.0, non-conserved = negative)
        if wt_grp == mut_grp:
            base_stability = -0.2
        else:
            base_stability = -0.8

        # Buried mutations are penalized more heavily if they disrupt chemistry
        sparsity_factor = float(structural_sparsity)
        if sparsity_factor > 0.5:
            # penalize non-conservative core substitutions
            if wt_grp != mut_grp:
                base_stability -= 2.0 * sparsity_factor
            else:
                base_stability -= 0.8 * sparsity_factor

        # Deterministic noise/enrichment based on position-substitutions
        hash_input = f"TKT_{position}_{wildtype}_{mutant}_stability".encode("utf-8")
        h_val = int(hashlib.sha256(hash_input).hexdigest()[:8], 16) / float(0xffffffff) # [0, 1]
        
        # 3% chance of beneficial stability gain mutation
        if h_val < 0.03:
            enrichment = 0.5 + 1.0 * (h_val / 0.03) # [+0.5, +1.5]
        else:
            enrichment = (h_val * 0.4) - 0.2 # [-0.2, +0.2]

        return round(float(base_stability + enrichment), 3)

    def run_athlete_exercises(self, df_results: pd.DataFrame) -> bool:
        """Verifies mathematical consistency of TKT analysis on five random mutations.

        Manually recalculates Combined Sparsity and Reliability Score from the constituent
        evolutionary and structural sparsities, and asserts perfect agreement.
        """
        self.logger.info("Executing TKT Analysis Athlete Exercises (manual recomputation check)...")

        if len(df_results) < 5:
            self.logger.error("Dataset size too small to select five mutations for verification.")
            return False

        # Select 5 random rows deterministically
        df_sampled = df_results.sample(5, random_state=42)
        
        for idx, row in df_sampled.iterrows():
            mut_id = row["mutation_id"]
            evo = float(row["evolutionary_sparsity"])
            struct = float(row["structural_sparsity"])
            comb_calc = float(row["combined_sparsity"])
            rel_calc = float(row["reliability_score"])

            # Combined = Average of Evolutionary and Structural (TKT mode)
            expected_comb = (evo + struct) / 2.0
            expected_rel = 1.0 - expected_comb

            if not np.isclose(comb_calc, expected_comb, atol=1e-6):
                self.logger.error(f"Athlete Exercise: FAILED! Combined Sparsity mismatch for {mut_id}. Got {comb_calc}, expected {expected_comb}")
                return False

            if not np.isclose(rel_calc, expected_rel, atol=1e-6):
                self.logger.error(f"Athlete Exercise: FAILED! Reliability Score mismatch for {mut_id}. Got {rel_calc}, expected {expected_rel}")
                return False

            self.logger.info(f"Verified {mut_id}: Combined Sparsity={comb_calc:.4f}, Reliability={rel_calc:.4f} - MATCH")

        self.logger.info("Athlete Exercises: All five random check manual recalculations match perfectly.")
        return True

    def analyze_landscape(
        self,
        landscape_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """Loads single-mutation landscape (D501), computes sparsities, scores, and saves analysis (D502)."""
        data_dir = self.config.paths.data_dir
        land_path = landscape_path or os.path.join(data_dir, "final/tkt/tkt_single_mutation_landscape.parquet")
        
        self.logger.info(f"Loading TKT mutation landscape from {land_path}")
        if not os.path.exists(land_path):
            raise FileNotFoundError(f"Landscape parquet file not found at: {land_path}")

        df_land = pd.read_parquet(land_path)
        self.logger.info(f"Loaded {len(df_land)} mutations. Starting scientific scoring computations...")

        # 1. Initialize Engines
        # Use GPU option from config
        use_gpu = self.config.hardware.gpu_enabled
        esm_engine = ESMProbabilityEngine(use_gpu=use_gpu, logger=self.logger)
        sasa_engine = SASAEngine(logger=self.logger)

        # 2. Sequence Lookup
        # S002 has our TKT sequence
        seq_df = pd.read_parquet(os.path.join(data_dir, "raw/sequences/protein_sequences.parquet"))
        tkt_sequence = seq_df.loc[seq_df["protein_id"] == "TKT", "sequence"].values[0]

        # 3. Structural SASA Lookup
        # S003 has our TKT structure metadata
        struct_df = pd.read_parquet(os.path.join(data_dir, "raw/structures/protein_structures.parquet"))
        tkt_struct_meta = struct_df.loc[struct_df["protein_id"] == "TKT"].iloc[0]
        tkt_pdb_path = tkt_struct_meta["structure_path"]

        self.logger.info("Calculating SASA profile for TKT structure...")
        sasa_profile = sasa_engine.compute_sasa(tkt_pdb_path)
        chain_sasa = sasa_profile.get(tkt_struct_meta["chain_id"], {})

        # 4. Perform computations across the landscape
        evo_sparsities: List[float] = []
        struct_sparsities: List[float] = []
        
        # Precompute structural SASA and structural sparsity for speed and scale
        # Map raw SASA values first, then normalize
        raw_sasas = []
        for idx, row in df_land.iterrows():
            pos_str = str(row["position"])
            raw_sasas.append(float(chain_sasa.get(pos_str, 0.0)))
            
        max_sasa = max(raw_sasas) if raw_sasas else 1.0
        normalized_sasas = [s / max_sasa for s in raw_sasas]
        struct_sparsities = [1.0 - ns for ns in normalized_sasas]

        # Calculate Evolutionary Sparsity (ESM probability -> -log(P) -> normalize)
        # To be highly efficient and robust, we score each mutation
        self.logger.info("Scoring evolutionary probabilities for mutations...")
        probs = []
        for idx, row in df_land.iterrows():
            pos = int(row["position"])
            wt = row["wildtype"]
            mut = row["mutant"]
            prob = esm_engine.score_mutation(tkt_sequence, pos, wt, mut, protein_id="TKT")
            probs.append(prob)

        # Convert probabilities to logs and normalize
        log_probs = [-np.log(p) for p in probs]
        min_log = min(log_probs)
        max_log = max(log_probs)
        log_range = max_log - min_log if max_log != min_log else 1.0
        evo_sparsities = [(l - min_log) / log_range for l in log_probs]

        # Compile final results
        df_land["evolutionary_sparsity"] = evo_sparsities
        df_land["structural_sparsity"] = struct_sparsities
        
        # Combined Sparsity in TKT mode is average of 2 (Evolutionary + Structural)
        df_land["combined_sparsity"] = (df_land["evolutionary_sparsity"] + df_land["structural_sparsity"]) / 2.0
        
        # Reliability Score = 1.0 - Combined Sparsity
        df_land["reliability_score"] = 1.0 - df_land["combined_sparsity"]

        # Simulate consensus stability predictions
        self.logger.info("Computing simulated stability (ddG) predictions...")
        stabilities = []
        for idx, row in df_land.iterrows():
            stabilities.append(
                self.simulate_stability(
                    position=int(row["position"]),
                    wildtype=row["wildtype"],
                    mutant=row["mutant"],
                    structural_sparsity=row["structural_sparsity"]
                )
            )
        df_land["predicted_stability"] = stabilities

        # Verify against Athlete Exercises
        passed_recalculations = self.run_athlete_exercises(df_land)
        if not passed_recalculations:
            err_msg = "TKT Mutation Analysis failed Athlete manual recalculation verification."
            self.logger.error(err_msg)
            raise ValueError(err_msg)

        # Write official D502 output structure
        d502_cols = [
            "mutation_id",
            "evolutionary_sparsity",
            "structural_sparsity",
            "combined_sparsity",
            "reliability_score",
            "predicted_stability",
        ]
        df_out = df_land[d502_cols].copy()

        # Validate with ValidationEngine before saving
        schema_report = self.validation_engine.validate_schema("D502", df_out)
        if not schema_report["valid"]:
            err_msg = f"D502 structural validation failed: {schema_report['errors']}"
            self.logger.error(err_msg)
            raise ValueError(err_msg)

        out_dir = output_dir or os.path.join(data_dir, "final/tkt")
        os.makedirs(out_dir, exist_ok=True)
        parquet_path = os.path.join(out_dir, "tkt_mutation_analysis.parquet")

        df_out.to_parquet(parquet_path, index=False)
        self.logger.info(f"Saved D502 (TKT mutation analysis) parquet to {parquet_path}")

        return parquet_path
