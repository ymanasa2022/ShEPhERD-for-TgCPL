"""
tgcpl -- shared logic for TgCPL-selective inhibitor generation with ShEPhERD-2.

Layout
------
  tgcpl.leads       reference compounds and registry-id lookup
  tgcpl.scaffolds   scaffold cores for --scaffold-constraint
  tgcpl.properties  rule-based descriptors and triage thresholds
  tgcpl.conformers  3D embedding (needs the shepherd stack -- NOT imported here)

The CLI entry points at the repo root import from this package:

  tgcpl_generate.py     generation driver
  filter_generated.py   post-generation triage -> passing.smi / passing.sdf
  summarize_run.py      reporting over a run directory
  baseline_scores.py    null-model comparison

conformers is left out of these re-exports on purpose: importing it pulls in
shepherd and torch, and filtering and reporting should keep working in a plain
RDKit environment. Import it directly when you need it:

    from tgcpl.conformers import embed_lead
"""

from __future__ import annotations

from tgcpl.leads import (
    LEADS,
    get_lead,
    lead_names,
    name_for_registry_id,
    registry_id,
)
from tgcpl.properties import (
    DEFAULT_THRESHOLDS,
    NITRILE,
    TRIAZINE,
    describe,
    max_ring_size,
    pains_catalog,
    reasons_to_reject,
    thresholds,
)
from tgcpl.scaffolds import (
    CORE_ATOM_COUNTS,
    DEFAULT_CORE,
    P1_FREE_CORES,
    SCAFFOLD_CORES,
    TRIAZINE_CORE_SMARTS,
    core_names,
    get_scaffold_atom_indices,
)

__all__ = [
    # leads
    "LEADS",
    "get_lead",
    "lead_names",
    "name_for_registry_id",
    "registry_id",
    # scaffolds
    "SCAFFOLD_CORES",
    "CORE_ATOM_COUNTS",
    "P1_FREE_CORES",
    "DEFAULT_CORE",
    "TRIAZINE_CORE_SMARTS",
    "core_names",
    "get_scaffold_atom_indices",
    # properties
    "DEFAULT_THRESHOLDS",
    "NITRILE",
    "TRIAZINE",
    "describe",
    "max_ring_size",
    "pains_catalog",
    "reasons_to_reject",
    "thresholds",
]
