"""3D conformer generation for reference leads.

The only module needing shepherd/torch. __init__ does not import it, so
filtering and reporting work in a plain RDKit environment.
"""

from __future__ import annotations

import sys

from rdkit import Chem
from rdkit.Chem import AllChem

from shepherd_score.conformer_generation import embed_conformer_from_smiles

# Distance geometry is stochastic: a flexible lead can fail at one seed and
# succeed at the next. Cpd12 returned None at 42 and killed an 8 h GPU job.
# Fixed ladder, tried in order, so a given lead still embeds reproducibly.
RETRY_SEEDS = (0, 1, 7, 123, 2024)


def _embed_rdkit(smiles: str, seed: int) -> Chem.Mol | None:
    """Fallback for when shepherd_score's embedder gives up. Same contract:
    explicit Hs, one MMFF-relaxed conformer."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.maxIterations = 2000
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None

    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    return mol


def embed_lead(smiles: str, seed: int = 42) -> Chem.Mol | None:
    """MMFF only; xTB relaxation and ESP charges come later, via
    extract_interaction_profile."""
    for s in (seed, *RETRY_SEEDS):
        try:
            mol = embed_conformer_from_smiles(
                smiles, MMFF_optimize=True, random_seed=s
            )
        except Exception as exc:
            print(f"  embed: seed {s} raised {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            continue
        if mol is not None and mol.GetNumConformers() > 0:
            if s != seed:
                print(f"  embed: seed {seed} failed, used {s}")
            return mol

    mol = _embed_rdkit(smiles, seed)
    if mol is not None:
        print("  embed: shepherd_score failed at every seed, used RDKit ETKDGv3")
    return mol
