#!/usr/bin/env python
"""
baseline_scores.py — give the 3D similarity scores a scale.

    python baseline_scores.py --ref Cpd18 --n-random 100

A shape Tanimoto of 0.60 means nothing on its own. This builds the two
reference points that make it readable:

  null     random MOSES-aq molecules scored against the reference. What an
           unrelated molecule gets. Conditioning has to beat this.
  ceiling  the other TgCPL leads scored against the reference. These are real
           actives from the same series, so they show what a genuine analogue
           of the reference scores — the realistic target, not 1.0.

Generated molecules then land as a percentile of the null rather than as a
bare number.

Alignment
---------
Everything is scored after optimal alignment to the reference. Generated
molecules come out already in the reference's frame, but MOSES molecules and
freshly embedded leads do not, and scoring those in place would give ~0 and
make the null flatteringly easy to beat. Aligning everything is the honest
comparison; it biases *against* the generated set, which is the right
direction for a sanity check.

Size matching
-------------
Shape Tanimoto rises with similarity in size, so an unmatched null measures
the size effect instead of conditioning. --size-tol keeps null molecules
within a heavy-atom window of the reference.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

from shepherd_score.container import Molecule, MoleculePair
from shepherd_score.score.constants import ALPHA

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tgcpl.conformers import embed_lead  # noqa: E402
from tgcpl.leads import LEADS  # noqa: E402

RDLogger.DisableLog("rdApp.*")

NUM_SURF_POINTS = 400
MOSES_PKL = Path("../shepherd2/data/conformers/moses_aq/moses_test_scaffold_molblock_charges.pkl")


def aligned_scores(ref_molec: Molecule, fit_mol: Chem.Mol, alpha: float,
                   num_repeats: int = 20, verbose: bool = False
                   ) -> tuple[float | None, float | None]:
    """
    Best-alignment shape and pharmacophore similarity of fit_mol to the reference.

    align_with_* returns the transformed coordinates; the Tanimoto it achieved is
    left on the pair as sim_aligned_surf / sim_aligned_pharm.
    """
    try:
        fit_molec = Molecule(
            fit_mol,
            num_surf_points=NUM_SURF_POINTS,
            pharm_multi_vector=False,
            probe_radius=ref_molec.probe_radius,
        )
        mp = MoleculePair(ref_molec, fit_molec, num_surf_points=NUM_SURF_POINTS)
        mp.align_with_surf(alpha=alpha, num_repeats=num_repeats)
        surf = float(np.asarray(mp.sim_aligned_surf).max())
    except Exception as e:
        if verbose:
            print(f"    (shape scoring failed: {type(e).__name__}: {e})")
        return None, None

    try:
        mp.align_with_pharm(num_repeats=num_repeats)
        pharm = float(np.asarray(mp.sim_aligned_pharm).max())
    except Exception as e:
        if verbose:
            print(f"    (pharm scoring failed: {type(e).__name__}: {e})")
        pharm = None
    return surf, pharm


def load_moses(path: Path, limit: int) -> list[Chem.Mol]:
    with open(path, "rb") as f:
        entries = pickle.load(f)
    mols = []
    for molblock, _charges in entries[:limit]:
        m = Chem.MolFromMolBlock(molblock, removeHs=False)
        if m is not None:
            mols.append(m)
    return mols


def describe(label: str, vals: list[float]) -> None:
    if not vals:
        print(f"  {label:<24} (none)")
        return
    a = np.array(vals)
    print(
        f"  {label:<24} n={len(a):<4} mean {a.mean():.3f}  sd {a.std():.3f}  "
        f"p50 {np.percentile(a,50):.3f}  p90 {np.percentile(a,90):.3f}  max {a.max():.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", default="Cpd18", choices=list(LEADS))
    ap.add_argument("--n-random", type=int, default=100)
    ap.add_argument("--size-tol", type=int, default=12,
                    help="Keep null molecules within +/- this many heavy atoms of the reference")
    ap.add_argument("--moses-pkl", type=Path, default=MOSES_PKL)
    ap.add_argument("--generated", type=Path, default=None,
                    help="Optional run directory to place against the baseline")
    ap.add_argument("--out", type=Path, default=Path("local_outputs/baseline.npz"))
    args = ap.parse_args()

    alpha = ALPHA(NUM_SURF_POINTS)

    print(f"Reference: {args.ref}")
    ref_mol = embed_lead(LEADS[args.ref]["smiles"])
    if ref_mol is None:
        raise SystemExit(f"Could not embed {args.ref}")
    ref_heavy = Chem.RemoveHs(ref_mol).GetNumHeavyAtoms()
    ref_molec = Molecule(ref_mol, num_surf_points=NUM_SURF_POINTS, pharm_multi_vector=False)
    print(f"  heavy atoms: {ref_heavy}\n")

    # --- ceiling: the other leads, real actives from the same series ---
    print("CEILING — other TgCPL leads vs reference")
    ceiling = []
    for name, lead in LEADS.items():
        if name == args.ref:
            continue
        # ETKDG can fail on a given seed for these flexible leads; retry a few.
        m = None
        for seed in (42, 0, 7, 123):
            m = embed_lead(lead["smiles"], seed=seed)
            if m is not None:
                break
        if m is None:
            print(f"  {name:<8} SKIPPED: could not embed after 4 seeds")
            continue
        s, p = aligned_scores(ref_molec, m, alpha, verbose=True)
        if s is None:
            print(f"  {name:<8} SKIPPED: scoring failed")
            continue
        ceiling.append(s)
        print(f"  {name:<8} shape {s:.3f}   pharm {'-' if p is None else f'{p:.3f}'}"
              f"   (TgCPL IC50 {lead['tgcpl_ic50_nM']} nM)")
    print()

    # --- null: size-matched random MOSES molecules ---
    print(f"NULL — random MOSES-aq molecules (+/- {args.size_tol} heavy atoms)")
    pool = load_moses(args.moses_pkl, args.n_random * 4)
    sized = [m for m in pool
             if abs(Chem.RemoveHs(m).GetNumHeavyAtoms() - ref_heavy) <= args.size_tol]
    print(f"  pool {len(pool)} -> size-matched {len(sized)}, scoring {min(len(sized), args.n_random)}")
    null_surf, null_pharm = [], []
    for i, m in enumerate(sized[: args.n_random], 1):
        s, p = aligned_scores(ref_molec, m, alpha)
        if s is not None:
            null_surf.append(s)
        if p is not None:
            null_pharm.append(p)
        if i % 10 == 0:
            print(f"    {i} scored...", flush=True)
    print()

    print("SUMMARY (optimally aligned)")
    describe("null shape", null_surf)
    describe("null pharm", null_pharm)
    describe("ceiling shape (leads)", ceiling)

    gen_surf = []
    if args.generated:
        ref_dir = args.generated / args.ref
        sdf = ref_dir / "generated_molecules.sdf"
        if sdf.exists():
            gen = [m for m in Chem.SDMolSupplier(str(sdf), removeHs=False) if m is not None]
            for m in gen:
                s, _ = aligned_scores(ref_molec, m, alpha)
                if s is not None:
                    gen_surf.append(s)
            describe("generated shape", gen_surf)
            if null_surf and gen_surf:
                null = np.array(null_surf)
                print("\n  generated molecules as a percentile of the null:")
                for i, s in enumerate(sorted(gen_surf, reverse=True), 1):
                    pct = (null < s).mean() * 100
                    print(f"    {i:<3} shape {s:.3f}  ->  {pct:5.1f}th percentile")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, null_surf=null_surf, null_pharm=null_pharm,
             ceiling=ceiling, generated=gen_surf, ref=args.ref)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
