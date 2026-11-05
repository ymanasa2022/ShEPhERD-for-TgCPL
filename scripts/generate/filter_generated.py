"""
filter_generated.py
===================
Post-generation triage for tgcpl_generate.py output.

Applies, in order:
  1. parse + dedupe by canonical SMILES
  2. MECHANISM  : nitrile warhead and triazine must survive (ShEPhERD conditions
                  on a 3D field, not a substructure, so it is free to drop them)
  3. SIZE       : MW ceiling
  4. FLEXIBILITY: rotatable-bond ceiling
  5. RINGS      : max ring size -- rejects macrocycles and other oversized rings
  6. PROPERTIES : rule-based developability filters (Lipinski / Veber)
  7. PAINS      : RDKit FilterCatalog assay-interference substructures

A NOTE ON "ADMET": everything here is rule-based physicochemistry computed by
RDKit -- MW, logP, TPSA, HBD/HBA, rotatable bonds, PAINS. These are developability
*proxies*, not ADMET predictions. Real absorption/clearance/hERG/microsomal
stability endpoints need a trained predictor (ADMET-AI, Schrodinger QikProp, or
the paper's own AutoML ensemble). Treat a pass here as "not obviously
disqualified", not as "good ADMET".

Usage:
  python filter_generated.py tgcpl_outputs/run_vmcc5199_mw400
  python filter_generated.py <dir> --max-mw 400 --max-rotb 7 --max-ring-size 7
  python filter_generated.py <dir> --no-pains --out passing.smi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rdkit import Chem, RDLogger

# scripts/generate/ is this file's directory; the tgcpl package sits beside it
# under scripts/. Put the parent on sys.path so `import tgcpl` resolves however
# this is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tgcpl.properties import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    describe,
    pains_catalog,
    reasons_to_reject,
)

RDLogger.DisableLog("rdApp.*")

_D = DEFAULT_THRESHOLDS


def collect_sdfs(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    found = sorted(target.rglob("generated_molecules.sdf"))
    return found


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("target", type=Path,
                    help="A tgcpl_generate.py --output-dir, or a single .sdf")
    ap.add_argument("--max-mw", type=float, default=_D["max_mw"])
    ap.add_argument("--max-rotb", type=int, default=_D["max_rotb"])
    ap.add_argument("--max-ring-size", type=int, default=_D["max_ring_size"],
                    help="Largest allowed ring. 7 keeps cycloheptyl (Cpd13) and "
                         "rejects macrocycles")
    ap.add_argument("--max-logp", type=float, default=_D["max_logp"])
    ap.add_argument("--max-hbd", type=int, default=_D["max_hbd"])
    ap.add_argument("--max-hba", type=int, default=_D["max_hba"])
    ap.add_argument("--max-tpsa", type=float, default=_D["max_tpsa"],
                    help="Veber oral-absorption ceiling")
    ap.add_argument("--no-warhead-check", dest="require_warhead",
                    action="store_false")
    ap.add_argument("--no-triazine-check", dest="require_triazine",
                    action="store_false")
    ap.add_argument("--no-pains", dest="pains", action="store_false")
    ap.add_argument("--out", type=Path, default=None,
                    help="Write passing SMILES here (default: <target>/passing.smi)")
    ap.add_argument("--out-sdf", type=Path, default=None,
                    help="Write passing molecules WITH 3D coordinates here "
                         "(default: <target>/passing.sdf). This is the file to "
                         "hand to Boltz-2 -- the conformers are already "
                         "xTB-relaxed, so they do not need re-embedding.")
    args = ap.parse_args()

    sdfs = collect_sdfs(args.target)
    if not sdfs:
        sys.exit(f"No generated_molecules.sdf found under {args.target}")

    pains = pains_catalog() if args.pains else None

    # ---- load + dedupe -----------------------------------------------------
    seen: dict[str, Chem.Mol] = {}
    n_raw = 0
    for sdf in sdfs:
        for mol in Chem.SDMolSupplier(str(sdf), removeHs=True):
            if mol is None:
                continue
            n_raw += 1
            try:
                smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
            except Exception:
                continue
            seen.setdefault(smi, mol)

    print(f"Sources      : {len(sdfs)} SDF file(s)")
    for s in sdfs:
        print(f"  {s}")
    print(f"Parsed       : {n_raw}")
    print(f"Unique       : {len(seen)}")
    print(f"\nThresholds   : MW<={args.max_mw:g}  rotB<={args.max_rotb}  "
          f"maxRing<={args.max_ring_size}  logP<={args.max_logp:g}  "
          f"HBD<={args.max_hbd}  HBA<={args.max_hba}  TPSA<={args.max_tpsa:g}"
          f"{'  +PAINS' if pains else ''}")

    # ---- filter ------------------------------------------------------------
    passing, rejected = [], []
    tally: dict[str, int] = {}
    for smi, mol in seen.items():
        d = describe(mol)
        why = reasons_to_reject(mol, d, args, pains)
        for w in why:
            tally[w] = tally.get(w, 0) + 1
        (passing if not why else rejected).append((smi, mol, d, why))

    print(f"\nPASS         : {len(passing)}/{len(seen)}")
    if tally:
        print("\nRejection reasons (molecules can fail several):")
        for reason, n in sorted(tally.items(), key=lambda kv: -kv[1]):
            print(f"  {reason:<16} {n:>4}")

    if passing:
        passing.sort(key=lambda t: -t[2]["qed"])
        print(f"\n  {'#':<4}{'MW':>7}{'logP':>7}{'TPSA':>7}{'rotB':>6}"
              f"{'ring':>6}{'QED':>6}{'fsp3':>6}  SMILES")
        print(f"  {'-' * 92}")
        for i, (smi, _mol, d, _) in enumerate(passing, 1):
            shown = smi if len(smi) <= 44 else smi[:41] + "..."
            print(f"  {i:<4}{d['mw']:>7.1f}{d['logp']:>7.2f}{d['tpsa']:>7.1f}"
                  f"{d['rotb']:>6}{d['max_ring']:>6}{d['qed']:>6.2f}"
                  f"{d['fsp3']:>6.2f}  {shown}")

        base = args.target if args.target.is_dir() else args.target.parent
        out = args.out or base / "passing.smi"
        with out.open("w") as fh:
            fh.write("# filtered by filter_generated.py\n")
            fh.write(f"# MW<={args.max_mw:g} rotB<={args.max_rotb} "
                     f"maxRing<={args.max_ring_size}\n")
            for smi, _mol, d, _ in passing:
                fh.write(f"{smi}\tMW={d['mw']:.1f}\tQED={d['qed']:.2f}\n")
        print(f"\nWrote {len(passing)} SMILES -> {out}")

        # 3D SDF for downstream covalent docking. Coordinates come straight
        # from generation + xTB relaxation, so Boltz-2 gets a real conformer
        # rather than a fresh RDKit embedding of the SMILES.
        out_sdf = args.out_sdf or base / "passing.sdf"
        w = Chem.SDWriter(str(out_sdf))
        for smi, mol, d, _ in passing:
            mol.SetProp("_Name", smi)
            mol.SetProp("MW", f"{d['mw']:.1f}")
            mol.SetProp("QED", f"{d['qed']:.2f}")
            mol.SetProp("rotB", str(d["rotb"]))
            w.write(mol)
        w.close()
        print(f"Wrote {len(passing)} 3D structures -> {out_sdf}  (for Boltz-2)")
    else:
        print("\nNothing passed. Loosen --max-mw or --max-rotb, or check the "
              "rejection tally above for which constraint is binding.")


if __name__ == "__main__":
    main()
