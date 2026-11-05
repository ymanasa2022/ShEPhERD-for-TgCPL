"""
Reference lead molecules for TgCPL-selective covalent inhibitors.

All are triazine-nitrile reversible covalent inhibitors targeting Cys31 (TgCPL)
/ Cys25 (HsCPL). The shared pharmacophore is:

    P1 = morpholine on the triazine
    P2 = basic amine chain off the exocyclic (acyclic) triazine nitrogen
    P3 = arylmethyl / cycloalkyl group on that same nitrogen

``vmcc`` is the compound-registry substance_id, so generated output can be
joined back to assay records without a separate lookup table.

Potency values come from the project's SAR write-up and may lag the raw assay
records; check those before using the numbers quantitatively.
"""

from __future__ import annotations

LEADS: dict[str, dict] = {
    "Cpd1": {
        "smiles": "CCN(CC)CCN(Cc1ccc(F)cc1)c1nc(C#N)nc(N2CCOCC2)n1",
        "vmcc": "VMCC-0000178",
        "tgcpl_ic50_nM": 320,
        "hscpl_ic50_nM": 2240,
        "selectivity": 7.0,
        "note": "Parent compound; no benzylic substituent",
    },
    "Cpd12": {
        "smiles": "CCN(CC)CCN(c1nc(C#N)nc(N2CCOCC2)n1)C(c1ccc(F)cc1)C1CCCCC1",
        "vmcc": "VMCC-0005199",
        "tgcpl_ic50_nM": 629,
        "hscpl_ic50_nM": 23245,
        "selectivity": 36.9,
        "note": "Cyclohexyl benzylic; best single-SAR compound",
    },
    "Cpd13": {
        "smiles": "CCN(CC)CCN(c1nc(C#N)nc(N2CCOCC2)n1)C(c1ccc(F)cc1)C1CCCCCC1",
        "vmcc": "VMCC-0005223",
        "tgcpl_ic50_nM": 497,
        "hscpl_ic50_nM": 19695,
        "selectivity": 39.6,
        "note": "Cycloheptyl benzylic; best selectivity overall",
    },
    "Cpd18": {
        "smiles": "N#Cc1nc(N2CCOCC2)nc(N(CCC2CCNCC2)C(c2ccc(F)cc2)C2CCCCC2)n1",
        "vmcc": "VMCC-0005222",
        "tgcpl_ic50_nM": 210,
        "hscpl_ic50_nM": 8080,
        "selectivity": 38.5,
        "note": "Best potency + high selectivity; piperidinylethyl P2",
    },
}


def lead_names() -> list[str]:
    """Names accepted by --ref-smiles, in declaration order."""
    return list(LEADS)


def get_lead(name: str) -> dict:
    """Look up a lead by name, raising a message that lists the valid ones."""
    try:
        return LEADS[name]
    except KeyError:
        raise KeyError(
            f"unknown lead {name!r}; expected one of {', '.join(LEADS)}"
        ) from None


def registry_id(name: str) -> str:
    """Registry substance_id for a lead, or the name itself if none is recorded.

    Used to name run directories, so it must always return something usable as
    a path component rather than raising on an incomplete record.
    """
    return get_lead(name).get("vmcc") or name


def name_for_registry_id(substance_id: str) -> str | None:
    """Inverse of registry_id: registry substance_id -> internal lead name."""
    for name, rec in LEADS.items():
        if rec.get("vmcc") == substance_id:
            return name
    return None
