"""
3D conformer generation for reference leads.

Kept apart from the rest of the package because this is the only module that
needs the shepherd stack installed. tgcpl/__init__.py deliberately does NOT
import it, so filtering and reporting still work in a plain RDKit environment
with no torch and no shepherd -- which is what makes it possible to triage
results on a laptop while generation runs on a cluster.
"""

from __future__ import annotations

from rdkit import Chem

from shepherd_score.conformer_generation import embed_conformer_from_smiles


def embed_lead(smiles: str, seed: int = 42) -> Chem.Mol | None:
    """Embed lead SMILES as a 3D conformer with explicit H.

    MMFF-optimized here; the xTB relaxation and partial charges that feed the
    ESP come later, in extract_interaction_profile.
    """
    return embed_conformer_from_smiles(
        smiles, MMFF_optimize=True, random_seed=seed
    )
