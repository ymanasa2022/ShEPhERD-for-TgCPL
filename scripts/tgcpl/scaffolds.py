"""
Scaffold cores for --scaffold-constraint: which atoms stay fixed during
generation, expressed as SMARTS matched against the reference conformer.

ShEPhERD conditions on a 3D field, not a substructure, so nothing otherwise
stops it from discarding the warhead. Pinning atoms is the only hard structural
guarantee available, which is what these cores are for.

Cores, smallest first:

  warhead                  8 atoms.  Nitrile + triazine only: the
                           mechanistically required part, since the C#N is what
                           reacts with the catalytic cysteine. Everything else,
                           including P1, is rebuilt, so S1 is explorable.

  warhead_p2n              9 atoms.  Adds the acyclic P2 nitrogen, so the
                           attachment point stays put while P1 and BOTH groups
                           on that nitrogen are regenerated.

  warhead_morpholine      14 atoms.  Warhead + P1 morpholine.

  warhead_morpholine_p2n  15 atoms.  Warhead + P1 morpholine + P2 nitrogen;
                           only the two groups on that nitrogen are
                           regenerated. Pairs with warhead_p2n to isolate the
                           effect of replacing P1.

Reactivity caveat for the cores that free P1: the nitrile's electrophilicity
depends on how electron-poor the triazine is, which the P1 substituent tunes.
Replacing morpholine changes covalent reactivity, so analogues from those runs
need a reactivity check, not just a shape or docking score.

The [N;!R] in the p2n cores is load-bearing. Written as plain N the pattern
matches these leads TWICE, because the exocyclic N can map to either the P2
nitrogen or the morpholine nitrogen, and get_scaffold_atom_indices takes
matches[0] -- the morpholine one, pinning exactly the group a P1 sweep means to
vary. Requiring not-in-ring selects the acyclic P2 nitrogen uniquely.
"""

from __future__ import annotations

from rdkit import Chem

SCAFFOLD_CORES: dict[str, str] = {
    "warhead": "N#Cc1ncncn1",
    "warhead_p2n": "N#Cc1ncnc([N;!R])n1",
    "warhead_morpholine": "N#Cc1nc(N2CCOCC2)ncn1",
    "warhead_morpholine_p2n": "N#Cc1nc(N2CCOCC2)nc([N;!R])n1",
}

#: Expected pinned-atom count per core, used by the pre-flight test. A mismatch
#: means a SMARTS matched something other than what its name claims.
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
    """Atom indices in ``mol`` matching the chosen scaffold core.

    Returns [] when the core does not match, which callers treat as "run
    unconstrained" rather than an error.
    """
    core_query = Chem.MolFromSmarts(SCAFFOLD_CORES[core])
    if core_query is None:
        return []
    matches = mol.GetSubstructMatches(core_query, useChirality=False)
    if not matches:
        return []
    # Flatten all matches (take first if multiple)
    return list(matches[0])
