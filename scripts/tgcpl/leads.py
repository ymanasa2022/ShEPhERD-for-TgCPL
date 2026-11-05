"""
Reference lead molecules for TgCPL-selective covalent inhibitors.

All are triazine-nitrile reversible covalent inhibitors targeting Cys31 (TgCPL)
/ Cys25 (HsCPL). The shared pharmacophore is:

    P1 = morpholine on the triazine
    P2 = basic amine chain off the exocyclic (acyclic) triazine nitrogen
    P3 = arylmethyl / cycloalkyl group on that same nitrogen

``vmcc`` is the compound-registry substance_id, so generated output can be
joined back to assay records without a separate lookup table.

Potency = experimental readouts, 10 ** mean_log_ic50(uM) * 1000, matched by
InChIKey. NOT the SAR write-up's figures, which differ for 3 of 4 leads
(Cpd1: 7.0x there vs 16.2x here). Cpd1 is the --dual-target anti-target, so
the gap narrows that run's premise.
"""

from __future__ import annotations

LEADS: dict[str, dict] = {
    "Cpd1": {
        "smiles": "CCN(CC)CCN(Cc1ccc(F)cc1)c1nc(C#N)nc(N2CCOCC2)n1",
        "vmcc": "VMCC-0000178",
        "tgcpl_ic50_nM": 222.0,
        "hscpl_ic50_nM": 3600.6,
        "selectivity": 16.22,
        "note": "Parent compound; no benzylic substituent",
    },
    "Cpd12": {
        "smiles": "CCN(CC)CCN(c1nc(C#N)nc(N2CCOCC2)n1)C(c1ccc(F)cc1)C1CCCCC1",
        "vmcc": "VMCC-0005199",
        "tgcpl_ic50_nM": 557.6,
        "hscpl_ic50_nM": 18515.0,
        "selectivity": 33.21,
        "note": "Cyclohexyl benzylic; best single-SAR compound",
    },
    "Cpd13": {
        "smiles": "CCN(CC)CCN(c1nc(C#N)nc(N2CCOCC2)n1)C(c1ccc(F)cc1)C1CCCCCC1",
        "vmcc": "VMCC-0005223",
        "tgcpl_ic50_nM": 496.8,
        "hscpl_ic50_nM": 19695.0,
        "selectivity": 39.65,
        "note": "Cycloheptyl benzylic; best selectivity overall",
    },
    "Cpd18": {
        "smiles": "N#Cc1nc(N2CCOCC2)nc(N(CCC2CCNCC2)C(c2ccc(F)cc2)C2CCCCC2)n1",
        "vmcc": "VMCC-0005222",
        "tgcpl_ic50_nM": 207.1,
        "hscpl_ic50_nM": 6845.5,
        "selectivity": 33.05,
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
    """Registry id, or the name if none recorded. Always path-safe: it names
    run directories, so it must not raise on an incomplete record."""
    return get_lead(name).get("vmcc") or name


def name_for_registry_id(substance_id: str) -> str | None:
    """Inverse of registry_id: registry substance_id -> internal lead name."""
    for name, rec in LEADS.items():
        if rec.get("vmcc") == substance_id:
            return name
    return None
