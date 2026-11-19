"""Shared logic for TgCPL-selective inhibitor generation with ShEPhERD-2.

conformers is deliberately not re-exported: importing it pulls in shepherd and
torch. Use `from tgcpl.conformers import embed_lead`.
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
from tgcpl.selectivity import (
    SELECTIVE_LIKE_CUTOFF,
    WEIGHTS,
    benzylic_ring_has_heteroatom,
    benzylic_ring_sizes,
    describe_selectivity,
    selectivity_flags,
    selectivity_score,
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
    # selectivity
    "SELECTIVE_LIKE_CUTOFF",
    "WEIGHTS",
    "benzylic_ring_has_heteroatom",
    "benzylic_ring_sizes",
    "describe_selectivity",
    "selectivity_flags",
    "selectivity_score",
]
