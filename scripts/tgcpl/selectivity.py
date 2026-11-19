"""TgCPL-over-HsCPL selectivity heuristics as RDKit substructure rules.

NOT a model: the weights are hand-set from FOUR compounds, nothing was fit or
validated. They reward resemblance to the one chemotype known to work, so they
penalise the novel scaffolds generation exists to find. Triage only -- the
per-rule flags matter more than the total, and scripts/dock/ is the real
selectivity filter.
"""

from __future__ import annotations

from rdkit import Chem

# Covalent warhead. Not selectivity, but a molecule without it is not in the
# series at all, so it cannot be scored on this basis.
NITRILE = Chem.MolFromSmarts("C#N")
TRIAZINE = Chem.MolFromSmarts("c1ncncn1")

# The Cpd12/13/18 benzylic motif: one H on the benzylic carbon, a ring off it.
# Whether that ring is all-carbon is checked in Python, not here -- the THP
# oxygen sits three bonds from the attachment point, out of SMARTS reach.
BENZYLIC_CYCLOALKYL = Chem.MolFromSmarts("[c]-[CX4H1](-[NX3])-[CX4;R]")

# Cpd1: two benzylic hydrogens, so the aryl rotates into either enzyme's deep
# pocket. The least selective lead.
BENZYLIC_UNSUBSTITUTED = Chem.MolFromSmarts("[c]-[CH2]-[NX3]")

# gem-disubstituted benzylic carbon: SAR says this costs potency and selectivity.
BENZYLIC_QUATERNARY = Chem.MolFromSmarts("[c]-[CX4H0](-[NX3])")

# The base that has to reach Glu75/Asp218. !$(N-a) keeps the triazine N and
# anilines out.
BASIC_TERTIARY_AMINE = Chem.MolFromSmarts(
    "[NX3;H0;!$(N-a);!$(N-[C,S]=[O,S,N]);!$(N-N)]([CX4])([CX4])[CX4]"
)
# Cpd18's piperidine NH: donor as well as base, which is why it engages the
# acidic residues better than the diethylamines.
BASIC_SECONDARY_AMINE = Chem.MolFromSmarts(
    "[NX3;H1;!$(N-a);!$(N-[C,S]=[O,S,N]);!$(N-N)]([CX4])[CX4]"
)

# The THP analogue killed potency, so O/N/S anywhere in the benzylic ring is a
# known-bad substitution.
POLAR_RING_ELEMENTS = frozenset({"O", "N", "S"})

# Cyclohexyl (6) and cycloheptyl (7) are the two best leads; nothing larger
# exists in the series.
BENZYLIC_RING_MIN = 5
BENZYLIC_RING_MAX = 7

# Hand-set, not fit. Positive rewards the selective chemotype, negative flags
# a known-bad feature.
WEIGHTS: dict[str, float] = {
    "benzylic_bulk": 2.0,        # the main selectivity handle
    "benzylic_ring_ok": 1.0,     # and it has to be the right size
    "basic_amine": 1.5,          # must reach the acidic S2 residues
    "secondary_amine_donor": 0.5,  # Cpd18's edge over the diethylamines
    "benzylic_unsubstituted": -2.0,
    "benzylic_quaternary": -1.5,
    "benzylic_polar_ring": -1.5,
}

# Set so Cpd12/13/18 clear it and Cpd1 does not: a cut on the training SAR,
# not a validated threshold.
SELECTIVE_LIKE_CUTOFF = 3.0


def _carbocycle_sizes_at(mol: Chem.Mol, atom_idx: int) -> list[int]:
    sizes = []
    for ring in mol.GetRingInfo().AtomRings():
        if atom_idx not in ring:
            continue
        if all(mol.GetAtomWithIdx(i).GetSymbol() == "C" for i in ring):
            sizes.append(len(ring))
    return sizes


def _benzylic_ring_atoms(mol: Chem.Mol) -> list[int]:
    # match is (aryl_c, benzylic_c, amine_n, ring_c); the ring atom is last.
    return [m[3] for m in mol.GetSubstructMatches(BENZYLIC_CYCLOALKYL)]


def benzylic_ring_sizes(mol: Chem.Mol) -> list[int]:
    """Empty means no benzylic cycloalkyl, or a non-carbocyclic ring there --
    benzylic_ring_has_heteroatom() tells those two apart."""
    sizes: list[int] = []
    for idx in _benzylic_ring_atoms(mol):
        sizes.extend(_carbocycle_sizes_at(mol, idx))
    return sorted(set(sizes))


def benzylic_ring_has_heteroatom(mol: Chem.Mol) -> bool:
    """The THP case. Walks the ring rather than matching SMARTS because the
    heteroatom can sit anywhere in it -- in THP it is three bonds from the
    attachment carbon, out of SMARTS reach."""
    ring_info = mol.GetRingInfo()
    for idx in _benzylic_ring_atoms(mol):
        for ring in ring_info.AtomRings():
            if idx not in ring:
                continue
            if any(mol.GetAtomWithIdx(i).GetSymbol() in POLAR_RING_ELEMENTS
                   for i in ring):
                return True
    return False


def selectivity_flags(mol: Chem.Mol) -> dict[str, bool]:
    ring_sizes = benzylic_ring_sizes(mol)
    has_bulk = bool(mol.GetSubstructMatches(BENZYLIC_CYCLOALKYL))
    return {
        "benzylic_bulk": has_bulk,
        "benzylic_ring_ok": any(
            BENZYLIC_RING_MIN <= n <= BENZYLIC_RING_MAX for n in ring_sizes
        ),
        "basic_amine": mol.HasSubstructMatch(BASIC_TERTIARY_AMINE)
        or mol.HasSubstructMatch(BASIC_SECONDARY_AMINE),
        "secondary_amine_donor": mol.HasSubstructMatch(BASIC_SECONDARY_AMINE),
        # Only a penalty when there is no bulky benzylic centre anywhere: a
        # molecule can carry both motifs, and the good one is what counts.
        "benzylic_unsubstituted": mol.HasSubstructMatch(BENZYLIC_UNSUBSTITUTED)
        and not has_bulk,
        "benzylic_quaternary": mol.HasSubstructMatch(BENZYLIC_QUATERNARY),
        "benzylic_polar_ring": benzylic_ring_has_heteroatom(mol),
    }


def selectivity_score(mol: Chem.Mol) -> float:
    """Weighted sum of the rule flags; roughly -5 to +5."""
    flags = selectivity_flags(mol)
    return sum(WEIGHTS[k] for k, hit in flags.items() if hit)


def describe_selectivity(mol: Chem.Mol) -> dict:
    flags = selectivity_flags(mol)
    ring_sizes = benzylic_ring_sizes(mol)
    score = sum(WEIGHTS[k] for k, hit in flags.items() if hit)
    return {
        "sel_score": score,
        "selective_like": score >= SELECTIVE_LIKE_CUTOFF,
        "benzylic_rings": ring_sizes,
        "in_series": mol.HasSubstructMatch(NITRILE)
        and mol.HasSubstructMatch(TRIAZINE),
        # Rendered as a compact column: "+bulk +amine -quaternary".
        "sel_flags": " ".join(
            ("+" if WEIGHTS[k] > 0 else "-") + k.replace("benzylic_", "")
            for k, hit in flags.items()
            if hit
        ),
        **flags,
    }
