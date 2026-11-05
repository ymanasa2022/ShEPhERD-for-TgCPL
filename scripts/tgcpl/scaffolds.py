"""
Scaffold cores for --scaffold-constraint: which atoms stay fixed during
generation, expressed as SMARTS matched against the reference conformer.

ShEPhERD conditions on a 3D field, not a substructure, so pinning atoms is the
only hard structural guarantee that the warhead survives.

  warhead                  8  nitrile + triazine; P1 and P2 free
  warhead_p2n              9  + P2 nitrogen, so its two groups regrow in place
  warhead_morpholine      14  + P1 morpholine
  warhead_morpholine_p2n  15  both; pairs with warhead_p2n to isolate P1

Freeing P1 changes the nitrile's electrophilicity, so those runs need a
reactivity check, not just a docking score.

[N;!R] is load-bearing: plain N matches twice (P2 or morpholine nitrogen) and
get_scaffold_atom_indices takes matches[0], pinning the wrong one.
"""

from __future__ import annotations

from rdkit import Chem

SCAFFOLD_CORES: dict[str, str] = {
    "warhead": "N#Cc1ncncn1",
    "warhead_p2n": "N#Cc1ncnc([N;!R])n1",
    "warhead_morpholine": "N#Cc1nc(N2CCOCC2)ncn1",
    "warhead_morpholine_p2n": "N#Cc1nc(N2CCOCC2)nc([N;!R])n1",
}

#: Pinned-atom count per core; the pre-flight test asserts against this.
CORE_ATOM_COUNTS: dict[str, int] = {
    "warhead": 8,
    "warhead_p2n": 9,
    "warhead_morpholine": 14,
    "warhead_morpholine_p2n": 15,
}

#: Cores that leave the P1 substituent free to be regenerated.
P1_FREE_CORES = frozenset({"warhead", "warhead_p2n"})

# Back-compat alias; previously the only core.
TRIAZINE_CORE_SMARTS = SCAFFOLD_CORES["warhead_morpholine"]

DEFAULT_CORE = "warhead_morpholine"


def core_names() -> list[str]:
    """Names accepted by --scaffold-core."""
    return list(SCAFFOLD_CORES)


def get_scaffold_atom_indices(
    mol: Chem.Mol, core: str = DEFAULT_CORE
) -> list[int]:
    """Atom indices matching the core; [] means run unconstrained."""
    core_query = Chem.MolFromSmarts(SCAFFOLD_CORES[core])
    if core_query is None:
        return []
    matches = mol.GetSubstructMatches(core_query, useChirality=False)
    if not matches:
        return []
    # Flatten all matches (take first if multiple)
    return list(matches[0])
