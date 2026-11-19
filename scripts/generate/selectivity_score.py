"""
Rank generated molecules by predicted TgCPL-over-HsCPL selectivity, before
spending docking on them. Writes selectivity.tsv, and selected.sdf with --top.

Two signals, side by side rather than blended: substructure rules encoding the
series SAR, and (with --3d) dS = S_tg - S_hs from aligned ShEPhERD-score
similarity. ESP matters most, since the TgCPL/HsCPL S2 difference is
electrostatic (Glu75/Asp218 vs Leu69/Ala214), not steric.

The rules are circular -- hand-set from four compounds, so they reward
resemblance to the one chemotype already known to work. Neither signal sees a
protein, and the anti-target reference Cpd1 binds TgCPL too (16x selective),
so "unlike Cpd1" is a weak proxy for "does not fit HsCPL". This is a triage
order, not a selectivity prediction. --hs-ref-smi takes real human
cathepsin-L actives, which makes S_hs mean considerably more.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tgcpl lives under scripts/

from tgcpl.leads import LEADS  # noqa: E402
from tgcpl.properties import describe  # noqa: E402
from tgcpl.selectivity import describe_selectivity  # noqa: E402

RDLogger.DisableLog("rdApp.*")

NUM_SURF_POINTS = 400

#: Leads with real selectivity.
DEFAULT_TG_REFS = ["Cpd18", "Cpd12"]
#: Anti-target: the series parent, the only lead with substantial HsCPL
#: activity (3600 nM vs 18-20 uM for the others).
DEFAULT_HS_REFS = ["Cpd1"]


def collect_sdfs(target: Path) -> list[Path]:
    """passing.sdf if filtering has run, else the raw generation output."""
    if target.is_file():
        return [target]
    passing = sorted(target.rglob("passing.sdf"))
    if passing:
        return passing
    return sorted(target.rglob("generated_molecules.sdf"))


def load_unique(sdfs: list[Path]) -> dict[str, Chem.Mol]:
    """Canonical SMILES -> first molecule with it, 3D kept."""
    seen: dict[str, Chem.Mol] = {}
    for sdf in sdfs:
        for mol in Chem.SDMolSupplier(str(sdf), removeHs=False):
            if mol is None:
                continue
            try:
                smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
            except Exception:
                continue
            seen.setdefault(smi, mol)
    return seen


def _build_3d_scorer(tg_names, hs_names, hs_ref_smi, num_repeats, seed):
    """Imports live in here so the rules-only path runs in a plain RDKit env."""
    import numpy as np
    from shepherd_score.container import Molecule, MoleculePair
    from shepherd_score.score.constants import ALPHA

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tgcpl.conformers import embed_lead

    alpha = ALPHA(NUM_SURF_POINTS)

    def embed(smiles: str, label: str):
        # ETKDG fails on a given seed for these flexible leads; retry a few.
        for s in (seed, 0, 7, 123):
            m = embed_lead(smiles, seed=s)
            if m is not None:
                return m
        print(f"  WARNING: could not embed {label}; dropping it", file=sys.stderr)
        return None

    def as_ref(smiles, label):
        m = embed(smiles, label)
        if m is None:
            return None
        return Molecule(m, num_surf_points=NUM_SURF_POINTS, pharm_multi_vector=False)

    tg_refs = [(n, as_ref(LEADS[n]["smiles"], n)) for n in tg_names]
    hs_refs = [(n, as_ref(LEADS[n]["smiles"], n)) for n in hs_names]
    if hs_ref_smi:
        for i, line in enumerate(Path(hs_ref_smi).read_text().splitlines()):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            smi = line.split()[0]
            label = f"hs{i}"
            hs_refs.append((label, as_ref(smi, label)))

    tg_refs = [(n, r) for n, r in tg_refs if r is not None]
    hs_refs = [(n, r) for n, r in hs_refs if r is not None]
    if not tg_refs or not hs_refs:
        raise SystemExit("need at least one usable reference on each side")
    print(f"  TgCPL refs: {', '.join(n for n, _ in tg_refs)}")
    print(f"  HsCPL refs: {', '.join(n for n, _ in hs_refs)}")

    def pair_scores(ref_molec, fit_mol):
        try:
            fit = Molecule(
                fit_mol,
                num_surf_points=NUM_SURF_POINTS,
                pharm_multi_vector=False,
                probe_radius=ref_molec.probe_radius,
            )
            mp = MoleculePair(ref_molec, fit, num_surf_points=NUM_SURF_POINTS)
        except Exception:
            return None, None
        try:
            mp.align_with_esp(alpha=alpha, num_repeats=num_repeats)
            esp = float(np.asarray(mp.sim_aligned_esp).max())
        except Exception:
            esp = None
        try:
            mp.align_with_surf(alpha=alpha, num_repeats=num_repeats)
            surf = float(np.asarray(mp.sim_aligned_surf).max())
        except Exception:
            surf = None
        return esp, surf

    def best(refs, fit_mol):
        esps, surfs = [], []
        for _name, r in refs:
            e, s = pair_scores(r, fit_mol)
            if e is not None:
                esps.append(e)
            if s is not None:
                surfs.append(s)
        return (max(esps) if esps else None, max(surfs) if surfs else None)

    def score(mol):
        tg_esp, tg_surf = best(tg_refs, mol)
        hs_esp, hs_surf = best(hs_refs, mol)
        out = {
            "tg_esp": tg_esp, "hs_esp": hs_esp,
            "tg_surf": tg_surf, "hs_surf": hs_surf,
            "d_esp": None, "d_surf": None,
        }
        if tg_esp is not None and hs_esp is not None:
            out["d_esp"] = tg_esp - hs_esp
        if tg_surf is not None and hs_surf is not None:
            out["d_surf"] = tg_surf - hs_surf
        return out

    return score


COLUMNS = [
    "rank", "sel_score", "selective_like", "d_esp", "tg_esp", "hs_esp",
    "d_surf", "tg_surf", "hs_surf", "mw", "qed", "rotb", "max_ring",
    "benzylic_rings", "in_series", "sel_flags", "smiles",
]


def fmt(v) -> str:
    if v is None:
        return "NA"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, list):
        return ",".join(str(x) for x in v) or "NA"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("target", type=Path,
                    help="A run directory (passing.sdf is preferred over raw "
                         "output if present), or a single .sdf")
    ap.add_argument("--3d", dest="three_d", action="store_true",
                    help="Also compute dS from ShEPhERD-score. Needs the "
                         "shepherd env; roughly a second per molecule per "
                         "reference")
    ap.add_argument("--tg-refs", nargs="+", default=DEFAULT_TG_REFS,
                    choices=list(LEADS),
                    help="Leads whose profile a selective molecule should match")
    ap.add_argument("--hs-refs", nargs="+", default=DEFAULT_HS_REFS,
                    choices=list(LEADS),
                    help="Anti-target leads: profiles to look UNLIKE")
    ap.add_argument("--hs-ref-smi", type=Path, default=None,
                    help="File of extra anti-target SMILES, one per line "
                         "(e.g. known human cathepsin-L actives). Strongly "
                         "recommended: it is what makes S_hs mean something")
    ap.add_argument("--rank-by", default="auto",
                    choices=["auto", "sel_score", "d_esp", "d_surf"],
                    help="auto = d_esp with --3d, sel_score without")
    ap.add_argument("--top", type=int, default=0,
                    help="Write the top N to selected.sdf for docking")
    ap.add_argument("--require-series", action="store_true",
                    help="Drop molecules that lost the nitrile or triazine")
    ap.add_argument("--num-repeats", type=int, default=10,
                    help="Alignment restarts per pair; raise for stability, "
                         "lower for speed")
    ap.add_argument("--limit", type=int, default=0,
                    help="Score only the first N molecules (smoke testing)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None,
                    help="Default: <target>/selectivity.tsv")
    args = ap.parse_args()

    sdfs = collect_sdfs(args.target)
    if not sdfs:
        sys.exit(f"No passing.sdf or generated_molecules.sdf under {args.target}")

    print(f"Sources : {len(sdfs)} SDF file(s)")
    for s in sdfs:
        print(f"  {s}")

    seen = load_unique(sdfs)
    print(f"Unique  : {len(seen)}")

    rows = []
    for smi, mol in seen.items():
        flat = Chem.RemoveHs(mol)
        rec = {"smiles": smi}
        rec.update(describe_selectivity(flat))
        rec.update(describe(flat))
        rec["_mol"] = mol
        rows.append(rec)

    if args.require_series:
        before = len(rows)
        rows = [r for r in rows if r["in_series"]]
        print(f"In series: {len(rows)}/{before} (nitrile + triazine survived)")

    if args.limit:
        rows = rows[: args.limit]
        print(f"Limited : {len(rows)}")

    rank_by = args.rank_by
    if rank_by == "auto":
        rank_by = "d_esp" if args.three_d else "sel_score"

    if args.three_d:
        print("\nBuilding 3D references (xTB-free MMFF embed + MMFF94 charges)...")
        score3d = _build_3d_scorer(
            args.tg_refs, args.hs_refs, args.hs_ref_smi,
            args.num_repeats, args.seed,
        )
        print(f"Scoring {len(rows)} molecules in 3D...")
        for i, r in enumerate(rows, 1):
            r.update(score3d(r["_mol"]))
            if i % 10 == 0:
                print(f"  {i}/{len(rows)}", flush=True)
    else:
        for r in rows:
            r.update(dict.fromkeys(
                ["d_esp", "tg_esp", "hs_esp", "d_surf", "tg_surf", "hs_surf"]
            ))
        if rank_by != "sel_score":
            sys.exit(f"--rank-by {rank_by} needs --3d")

    # None sorts last: a molecule whose alignment failed should not win.
    rows.sort(key=lambda r: (r.get(rank_by) is not None, r.get(rank_by) or 0),
              reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    base = args.target if args.target.is_dir() else args.target.parent
    out = args.out or base / "selectivity.tsv"
    with out.open("w") as fh:
        fh.write(f"# selectivity_score.py, ranked by {rank_by}\n")
        fh.write(f"# TgCPL refs: {','.join(args.tg_refs)}\t"
                 f"HsCPL refs: {','.join(args.hs_refs)}\n")
        fh.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            fh.write("\t".join(fmt(r.get(c)) for c in COLUMNS) + "\n")
    print(f"\nWrote {len(rows)} rows -> {out}")

    show = rows[: args.top or 20]
    print(f"\nTop {len(show)} by {rank_by}:")
    print(f"  {'#':<4}{'sel':>6}{'dESP':>8}{'dSurf':>8}{'MW':>7}{'QED':>6}  flags")
    print(f"  {'-' * 74}")
    for r in show:
        print(f"  {r['rank']:<4}{r['sel_score']:>+6.1f}"
              f"{fmt(r['d_esp']):>8}{fmt(r['d_surf']):>8}"
              f"{r['mw']:>7.1f}{r['qed']:>6.2f}  {r['sel_flags']}")

    if args.top:
        sel = base / "selected.sdf"
        w = Chem.SDWriter(str(sel))
        for r in rows[: args.top]:
            m = r["_mol"]
            m.SetProp("_Name", r["smiles"])
            for c in ("rank", "sel_score", "d_esp", "d_surf", "mw", "qed"):
                m.SetProp(c, fmt(r.get(c)))
            w.write(m)
        w.close()
        print(f"\nWrote top {args.top} -> {sel}")
        print("  Next: scripts/dock/boltz_prep.py to fold these against "
              "BOTH TgCPL and HsCPL.")


if __name__ == "__main__":
    main()
