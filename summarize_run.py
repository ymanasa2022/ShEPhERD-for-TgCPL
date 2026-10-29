#!/usr/bin/env python
"""
summarize_run.py — read a tgcpl_generate.py output directory and report what
came out of it.

    python summarize_run.py local_outputs/run_cpd18_n10

tgcpl_generate.py writes an SDF and a SMILES file per reference but no summary,
so this answers the first questions you have about a run: how many molecules are
unique, how close they stay to the lead, and whether they look drug-like.

Tanimoto here is Morgan/ECFP4 against the reference lead. High similarity means
the model stayed near the lead; low means it wandered. Neither is automatically
good, but a run that returns the reference over and over is not giving you
anything, and that is easy to miss by eye.
"""

import argparse
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity

RDLogger.DisableLog("rdApp.*")

# Lead SMILES live in tgcpl_generate.py; import rather than duplicate them.
from tgcpl_generate import LEADS

_FPGEN = GetMorganGenerator(radius=2, fpSize=2048)

NUM_SURF_POINTS = 400

# These leads are covalent inhibitors: the nitrile is the electrophile that
# reacts with the catalytic cysteine. A generated molecule that has dropped it
# is not a covalent inhibitor at all, however well it scores on shape, so
# warhead retention is the first thing to check in any unconstrained run.
WARHEAD_PATTERNS = {
    "nitrile": "C#N",
    "triazine": "c1ncncn1",
    "nitrile_on_triazine": "N#Cc1ncncn1",
}


def fp(mol):
    return _FPGEN.GetFingerprint(mol)


def score_3d(ref_mol_3d, gen_mols_3d, pharm_multi_vector: bool = False):
    """
    Shape / ESP / pharmacophore similarity of each generated molecule to the
    reference, scored in the pose as generated.

    This is the number that says whether conditioning worked. ShEPhERD generates
    into the reference's own frame, so no alignment is done here — the same
    choice ConditionalEval makes for its `sim_*_target` scores. Both reference
    and generated molecules come out centered at the origin, so the poses are
    comparable without alignment — worth re-checking if either side changes.

    Two reasons these read slightly low. Partial charges fall back to MMFF94
    rather than xTB, so ESP is approximate. And reference.sdf is the MMFF
    geometry, while generation conditioned on the xTB-relaxed profile, so the
    reference here is a near copy of the true target rather than the target itself.
    This is a screen for "is conditioning working", not a publication number.

    Returns a list of (surf, esp, pharm) tuples, any element possibly None.
    """
    from shepherd_score.container import Molecule
    from shepherd_score.score.constants import ALPHA, LAM_SCALING
    from shepherd_score.score.electrostatic_scoring_np import get_overlap_esp_np
    from shepherd_score.score.gaussian_overlap_np import get_overlap_np
    from shepherd_score.score.pharmacophore_scoring_np import get_overlap_pharm_np

    alpha = ALPHA(NUM_SURF_POINTS)
    lam_scaled = 0.3 * LAM_SCALING

    ref = Molecule(
        ref_mol_3d,
        num_surf_points=NUM_SURF_POINTS,
        pharm_multi_vector=pharm_multi_vector,
    )

    out = []
    for m in gen_mols_3d:
        try:
            gen = Molecule(
                m,
                num_surf_points=NUM_SURF_POINTS,
                pharm_multi_vector=pharm_multi_vector,
                probe_radius=ref.probe_radius,
            )
        except Exception:
            out.append((None, None, None))
            continue

        surf = esp = pharm = None
        if gen.surf_pos is not None and ref.surf_pos is not None:
            surf = float(get_overlap_np(gen.surf_pos, ref.surf_pos, alpha=alpha))
            if gen.surf_esp is not None and ref.surf_esp is not None:
                esp = float(
                    get_overlap_esp_np(
                        gen.surf_pos, ref.surf_pos,
                        gen.surf_esp, ref.surf_esp,
                        alpha=alpha, lam=lam_scaled,
                    )
                )
        if gen.pharm_ancs is not None and ref.pharm_ancs is not None:
            pharm = float(
                get_overlap_pharm_np(
                    gen.pharm_types, ref.pharm_types,
                    gen.pharm_ancs, ref.pharm_ancs,
                    gen.pharm_vecs, ref.pharm_vecs,
                    similarity="tanimoto", extended_points=False,
                )
            )
        out.append((surf, esp, pharm))
    return out


def summarize_ref(ref_dir: Path, do_3d: bool = True) -> dict | None:
    """Summarize one reference's subdirectory. Returns None if it has no SDF."""
    sdf = ref_dir / "generated_molecules.sdf"
    if not sdf.exists():
        return None

    # Keep hydrogens for 3D scoring; drop them for SMILES so the strings are
    # readable and two molecules differing only in H placement count as one.
    mols_3d = [m for m in Chem.SDMolSupplier(str(sdf), removeHs=False) if m is not None]
    mols = [Chem.RemoveHs(m) for m in mols_3d]
    if not mols:
        print(f"{ref_dir.name}: SDF present but no readable molecules")
        return None

    smis = [Chem.MolToSmiles(m) for m in mols]
    unique = sorted(set(smis))

    # One representative 3D conformer per unique SMILES, for the 3D scores.
    rep_3d = {}
    for smi, m3 in zip(smis, mols_3d):
        rep_3d.setdefault(smi, m3)

    ref_name = ref_dir.name
    ref_smi = LEADS.get(ref_name, {}).get("smiles")
    ref_mol = Chem.MolFromSmiles(ref_smi) if ref_smi else None

    sims = []
    if ref_mol is not None:
        gen_mols = [Chem.MolFromSmiles(s) for s in unique]
        gen_mols = [m for m in gen_mols if m is not None]
        if gen_mols:
            sims = BulkTanimotoSimilarity(fp(ref_mol), [fp(m) for m in gen_mols])

    # 3D similarity to the reference, in the pose as generated.
    scores_3d = {}
    ref_sdf = ref_dir / "reference.sdf"
    if do_3d and ref_sdf.exists():
        ref_3d = next(iter(Chem.SDMolSupplier(str(ref_sdf), removeHs=False)), None)
        if ref_3d is not None:
            try:
                vals = score_3d(ref_3d, [rep_3d[s] for s in unique])
                scores_3d = dict(zip(unique, vals))
            except Exception as e:
                print(f"  (3D scoring unavailable: {type(e).__name__}: {e})")

    print(f"\n{'=' * 70}")
    print(f"  {ref_name}")
    print(f"{'=' * 70}")
    print(f"  molecules in SDF : {len(mols)}")
    print(f"  unique SMILES    : {len(unique)}  ({len(unique) / len(mols):.0%} of total)")
    if ref_smi:
        print(f"  reference        : {ref_smi}")
    if sims:
        exact = sum(1 for s in sims if s >= 0.999)
        print(f"  Tanimoto to ref  : min {min(sims):.2f} | mean {sum(sims)/len(sims):.2f} | max {max(sims):.2f}")
        if exact:
            print(f"  *** {exact} of {len(unique)} unique molecules ARE the reference ***")

    print("\n  warhead retention (of unique molecules):")
    for label, sma in WARHEAD_PATTERNS.items():
        q = Chem.MolFromSmarts(sma)
        kept = sum(
            1 for s in unique
            if (m := Chem.MolFromSmiles(s)) is not None and m.HasSubstructMatch(q)
        )
        flag = "  <-- covalent mechanism lost" if label == "nitrile" and kept < len(unique) else ""
        print(f"    {label:<20} {kept:>3}/{len(unique)}{flag}")

    if scores_3d:
        def col(j):
            return [v[j] for v in scores_3d.values() if v[j] is not None]
        for j, label in enumerate(("shape", "ESP  ", "pharm")):
            c = col(j)
            if c:
                print(f"  3D {label} vs ref : min {min(c):.2f} | mean {sum(c)/len(c):.2f} | max {max(c):.2f}")

        # At shepherd_score's default lam (0.3 * LAM_SCALING ~ 62) the ESP damping
        # term exp(-dESP^2/lam) sits at ~0.997 for ESP magnitudes of order 1, which
        # is what these molecules give. Measured directly: flipping the sign of a
        # generated molecule's entire ESP moved the score by 0.001. So ESP here is
        # the shape score relabelled, not independent evidence. Say so rather than
        # printing a column that looks like a second opinion.
        pairs = [(v[0], v[1]) for v in scores_3d.values() if v[0] is not None and v[1] is not None]
        if pairs:
            gap = sum(abs(s - e) for s, e in pairs) / len(pairs)
            if gap < 0.02:
                print(
                    f"  NOTE: ESP tracks shape to within {gap:.3f} on average — at the default\n"
                    f"        lam the ESP term barely modulates the score. Treat the ESP column\n"
                    f"        as shape, not as independent evidence."
                )

    print(
        f"\n  {'#':<4}{'MW':>7}{'logP':>7}{'QED':>6}{'Tan':>6}"
        f"{'shape':>7}{'ESP':>6}{'pharm':>7}  SMILES"
    )
    print(f"  {'-' * 80}")
    order = sorted(range(len(unique)), key=lambda i: -sims[i]) if sims else range(len(unique))
    for rank, i in enumerate(order, 1):
        m = Chem.MolFromSmiles(unique[i])
        if m is None:
            print(f"  {rank:<4}{'unparseable':>26}  {unique[i]}")
            continue
        tan = f"{sims[i]:.2f}" if sims else "  -"
        s3 = scores_3d.get(unique[i], (None, None, None))
        f3 = [f"{v:.2f}" if v is not None else "-" for v in s3]
        smi = unique[i] if len(unique[i]) <= 40 else unique[i][:37] + "..."
        print(
            f"  {rank:<4}{Descriptors.MolWt(m):>7.1f}{Crippen.MolLogP(m):>7.2f}"
            f"{QED.qed(m):>6.2f}{tan:>6}{f3[0]:>7}{f3[1]:>6}{f3[2]:>7}  {smi}"
        )

    return {"ref": ref_name, "n": len(mols), "unique": len(unique)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output_dir", type=Path, help="A tgcpl_generate.py --output-dir")
    ap.add_argument(
        "--no-3d", action="store_true",
        help="Skip shape/ESP/pharmacophore scoring (2D properties only, much faster)",
    )
    args = ap.parse_args()

    if not args.output_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.output_dir}")

    # One subdirectory per reference; a single-ref run still nests one level.
    ref_dirs = sorted(d for d in args.output_dir.iterdir() if d.is_dir())
    if not ref_dirs:
        ref_dirs = [args.output_dir]

    results = [r for d in ref_dirs if (r := summarize_ref(d, do_3d=not args.no_3d))]
    if not results:
        raise SystemExit(f"No generated_molecules.sdf found under {args.output_dir}")

    if len(results) > 1:
        total = sum(r["n"] for r in results)
        uniq = sum(r["unique"] for r in results)
        print(f"\n{'=' * 70}")
        print(f"  {len(results)} references | {total} molecules | {uniq} unique")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
