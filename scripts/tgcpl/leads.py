"""Reference lead molecules, loaded from data/leads.tsv (TGCPL_LEADS overrides).

Potency columns are experimental readouts, NOT the SAR write-up's figures,
which differ for 3 of 4 leads (Cpd1: 7.0x there vs 16.2x here). Cpd1 is the
--dual-target anti-target, so that gap narrows the run's premise.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

DEFAULT_LEADS_PATH = Path(__file__).resolve().parents[2] / "data" / "leads.tsv"

_FLOAT_FIELDS = ("tgcpl_ic50_nM", "hscpl_ic50_nM", "selectivity")


def leads_path() -> Path:
    return Path(os.environ.get("TGCPL_LEADS") or DEFAULT_LEADS_PATH)


def load_leads(path: Path | None = None) -> dict[str, dict]:
    path = Path(path) if path is not None else leads_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"leads file not found: {path}\n"
            f"Expected it in the repo; set TGCPL_LEADS to use another. "
            f"Columns are in data/leads.example.tsv."
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
    return list(LEADS)


def get_lead(name: str) -> dict:
    try:
        return LEADS[name]
    except KeyError:
        raise KeyError(
            f"unknown lead {name!r}; expected one of {', '.join(LEADS)}"
        ) from None


def registry_id(name: str) -> str:
    # Falls back to the name: this names run directories, so it must not
    # raise on an incomplete record.
    return get_lead(name).get("vmcc") or name


def name_for_registry_id(substance_id: str) -> str | None:
    for name, rec in LEADS.items():
        if rec.get("vmcc") == substance_id:
            return name
    return None
