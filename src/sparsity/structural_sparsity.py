import freesasa
from typing import Dict, Any, List


class StructuralSparsity:
    """
    Computes structural sparsity using SASA values from FreeSASA.
    """

    def __init__(self, pdb_path: str):
        self.pdb_path = pdb_path
        self.structure = freesasa.Structure(pdb_path)
        self.result = freesasa.calc(self.structure)
        self.residue_sasa = self._compute_residue_sasa()

    def _compute_residue_sasa(self) -> Dict[int, float]:
        """
        Compute SASA for each residue index in the PDB structure.
        Returns a dict: {residue_index: sasa_value}
        """
        residue_sasa = {}
        for i in range(self.structure.nResidues()):
            residue_sasa[i] = self.result.residueAreas()[i].total
        return residue_sasa

    def sparsity_score(self, mutation: Dict[str, Any]) -> float:
        """
        Compute structural sparsity for a single mutation.
        Lower SASA → higher sparsity (buried residue).
        """
        pos = mutation["position"]
        sasa = self.residue_sasa.get(pos, None)

        if sasa is None:
            raise ValueError(f"No SASA value for residue position {pos}")

        # Normalize SASA to [0,1] range using a simple heuristic
        max_sasa = max(self.residue_sasa.values())
        normalized = sasa / max_sasa

        # Structural sparsity = 1 - normalized SASA
        return 1.0 - normalized

    def sparsity_scores(self, mutations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compute structural sparsity for a list of mutations.
        """
        results = []
        for m in mutations:
            score = self.sparsity_score(m)
            out = m.copy()
            out["structural_sparsity"] = score
            results.append(out)
        return results

