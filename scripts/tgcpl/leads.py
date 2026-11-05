"""
Reference lead molecules, loaded from a TSV that git does not track.

Structures and assay values stay out of the repo. Point TGCPL_LEADS at the
file, or leave it at the default TgCPL/leads.tsv. See TgCPL/leads.example.tsv
for the columns.

Shared pharmacophore: P1 morpholine on the triazine, P2 basic chain off the
acyclic triazine nitrogen, P3 arylmethyl/cycloalkyl on that same nitrogen.
``vmcc`` is the registry substance_id, so output joins back to assay records.

Potency = experimental readouts, 10 ** mean_log_ic50(uM) * 1000, matched by
InChIKey. NOT the SAR write-up's figures, which differ for 3 of 4 leads
(Cpd1: 7.0x there vs 16.2x here). Cpd1 is the --dual-target anti-target, so
the gap narrows that run's premise.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

#: TgCPL/leads.tsv, relative to the repo root (two levels up from this file).
DEFAULT_LEADS_PATH = Path(__file__).resolve().parents[2] / "TgCPL" / "leads.tsv"

_FLOAT_FIELDS = ("tgcpl_ic50_nM", "hscpl_ic50_nM", "selectivity")


def leads_path() -> Path:
    """The TSV to load; TGCPL_LEADS overrides the default."""
    return Path(os.environ.get("TGCPL_LEADS") or DEFAULT_LEADS_PATH)


def load_leads(path: Path | None = None) -> dict[str, dict]:
    """Parse the leads TSV. Blank and '#' lines are skipped."""
    path = Path(path) if path is not None else leads_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"leads file not found: {path}\n"
            f"It is gitignored, so a fresh clone has to be given one. Copy it "
            f"in, or set TGCPL_LEADS. Columns are in TgCPL/leads.example.tsv."
        )
    rows = [
        ln for ln in path.read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    out: dict[str, dict] = {}
    for row in csv.DictReader(rows, delimiter="\t"):
        name = (row.get("name") or "").strip()
        if not name:
            continue
        rec = {k: (v.strip() if isinstance(v, str) else v)
               for k, v in row.items() if k and k != "name"}
        for f in _FLOAT_FIELDS:
            if rec.get(f):
                rec[f] = float(rec[f])
        out[name] = rec
    if not out:
        raise ValueError(f"no lead rows parsed from {path}")
    return out


LEADS: dict[str, dict] = load_leads()


def lead_names() -> list[str]:
    """Names accepted by --ref-smiles, in file order."""
    return list(LEADS)


def get_lead(name: str) -> dict:
    """Look up a lead, with a message listing the valid names."""
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
