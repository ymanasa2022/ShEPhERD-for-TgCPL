"""
3D conformer generation for reference leads.

The only module needing the shepherd stack. __init__ does not import it, so
filtering and reporting work in a plain RDKit environment.
"""

from __future__ import annotations

from rdkit import Chem

from shepherd_score.conformer_generation import embed_conformer_from_smiles


def embed_lead(smiles: str, seed: int = 42) -> Chem.Mol | None:
    """Embed lead SMILES as a 3D conformer with explicit H (MMFF only; the
    xTB relaxation and ESP charges come later via extract_interaction_profile).
    """
    return embed_conformer_from_smiles(
        smiles, MMFF_optimize=True, random_seed=seed
    )
