"""
Rule-based molecular property calculation and triage.

These are developability proxies, NOT ADMET predictions -- real endpoints need
a trained predictor. Crippen logP also runs low on this chemotype, so a logP
ceiling here is softer than the same number elsewhere.

RDKit only: no torch, no shepherd.
"""

from __future__ import annotations

import argparse

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors as rdmd
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# Mechanism-critical: the nitrile is the covalent warhead.
NITRILE = Chem.MolFromSmarts("C#N")
TRIAZINE = Chem.MolFromSmarts("c1ncncn1")

#: Shared by the CLI and the tests. max_ring_size 7 from the series histogram
#: {5:1, 6:440, 7:5}: admits the real chemistry, rejects macrocycles.
DEFAULT_THRESHOLDS = {
    "max_mw": 500.0,
    "max_rotb": 10,
    "max_ring_size": 7,
    "max_logp": 5.0,
    "max_hbd": 5,
    "max_hba": 10,
    "max_tpsa": 140.0,
    "require_warhead": True,
    "require_triazine": True,
}


def pains_catalog() -> FilterCatalog:
    """RDKit PAINS assay-interference catalog."""
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)


def max_ring_size(mol: Chem.Mol) -> int:
    """Largest SSSR ring, 0 for an acyclic molecule."""
    return max((len(r) for r in mol.GetRingInfo().AtomRings()), default=0)


def describe(mol: Chem.Mol) -> dict:
    """Physicochemical descriptors for one molecule."""
    return {
        "mw": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "tpsa": rdmd.CalcTPSA(mol),
        "hbd": rdmd.CalcNumHBD(mol),
        "hba": rdmd.CalcNumHBA(mol),
        "rotb": rdmd.CalcNumRotatableBonds(mol),
        "qed": QED.qed(mol),
        "rings": rdmd.CalcNumRings(mol),
        "max_ring": max_ring_size(mol),
        "fsp3": rdmd.CalcFractionCSP3(mol),
        "heavy": mol.GetNumHeavyAtoms(),
    }


def thresholds(**overrides) -> argparse.Namespace:
    """DEFAULT_THRESHOLDS plus overrides, as the namespace argparse yields."""
    merged = dict(DEFAULT_THRESHOLDS)
    unknown = set(overrides) - set(merged)
    if unknown:
        raise TypeError(f"unknown threshold(s): {', '.join(sorted(unknown))}")
    merged.update(overrides)
    return argparse.Namespace(**merged)


def reasons_to_reject(
    mol: Chem.Mol,
    d: dict,
    limits: argparse.Namespace,
    pains: FilterCatalog | None = None,
) -> list[str]:
    """Why this molecule fails triage; [] means it passes. Returns every
    reason, so a tally shows which constraint is binding."""
    out = []
    if limits.require_warhead and not mol.HasSubstructMatch(NITRILE):
        out.append("no-nitrile")
    if limits.require_triazine and not mol.HasSubstructMatch(TRIAZINE):
        out.append("no-triazine")
    if d["mw"] > limits.max_mw:
        out.append(f"MW>{limits.max_mw:g}")
    if d["rotb"] > limits.max_rotb:
        out.append(f"rotB>{limits.max_rotb}")
    if d["max_ring"] > limits.max_ring_size:
        out.append(f"ring{d['max_ring']}>{limits.max_ring_size}")
    if d["logp"] > limits.max_logp:
        out.append(f"logP>{limits.max_logp:g}")
    if d["hbd"] > limits.max_hbd:
        out.append(f"HBD>{limits.max_hbd}")
    if d["hba"] > limits.max_hba:
        out.append(f"HBA>{limits.max_hba}")
    if d["tpsa"] > limits.max_tpsa:
        out.append(f"TPSA>{limits.max_tpsa:g}")
    if pains is not None and pains.HasMatch(mol):
        out.append("PAINS")
    return out
