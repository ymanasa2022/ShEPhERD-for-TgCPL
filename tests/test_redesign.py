"""
tests/test_redesign.py
======================
Pre-flight check for jobs that use the scaffold cores and property filter, so a
typo does not cost an 8-hour GPU job.

  python tests/test_redesign.py          # ~5 s, RDKit only -- no torch, no GPU
  python tests/test_redesign.py --full   # also runs a real 2-sample generation

Every check prints its expected value next to what it actually got. A clean run
ends with "ALL CHECKS PASSED" and exit code 0; any failure exits 1.

The quick path imports only tgcpl.leads / scaffolds / properties, which are
RDKit-only, so it runs in a plain environment. Only --full pulls in the
generation stack.

What it does NOT check: whether generation is numerically well-behaved at a
reduced N_x1. Only --full touches the model, and even then only lightly.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors as rdmd

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parent.parent
GENERATE = REPO / "scripts" / "generate"
TGCPL_PKG = REPO / "scripts" / "tgcpl"

# The package lives under scripts/, not at the repo root.
sys.path.insert(0, str(REPO / "scripts"))

from tgcpl import leads, properties, scaffolds          # noqa: E402

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, str, str]] = []


def check(name: str, got, expected, ok: bool | None = None) -> bool:
    good = (got == expected) if ok is None else ok
    _results.append((PASS if good else FAIL, name, f"expected {expected!r}, got {got!r}"))
    print(f"  [{PASS if good else FAIL}] {name}\n        expected {expected!r}, got {got!r}")
    return good


def section(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 62 - len(title)))


# =============================================================================
# 1-2. scaffold cores
# =============================================================================
def test_cores() -> None:
    section("1. scaffold cores pin the right atoms")

    check("core table matches declared atom counts",
          sorted(scaffolds.SCAFFOLD_CORES), sorted(scaffolds.CORE_ATOM_COUNTS))

    for lead in ("Cpd12", "Cpd18"):
        mol = Chem.AddHs(Chem.MolFromSmiles(leads.LEADS[lead]["smiles"]))
        for core, n_expected in scaffolds.CORE_ATOM_COUNTS.items():
            inds = scaffolds.get_scaffold_atom_indices(mol, core)
            check(f"{lead} / {core}: pinned atom count", len(inds), n_expected)

    # The ambiguity trap: the P2 nitrogen must be pinned, not the morpholine N.
    section("2. the pinned nitrogen is the P2 nitrogen, not morpholine's")
    for lead in ("Cpd12", "Cpd18"):
        mol = Chem.AddHs(Chem.MolFromSmiles(leads.LEADS[lead]["smiles"]))
        for core in ("warhead_p2n", "warhead_morpholine_p2n"):
            inds = scaffolds.get_scaffold_atom_indices(mol, core)
            # morpholine contributes exactly one O; its absence proves P1 is free
            want_o = 0 if core in scaffolds.P1_FREE_CORES else 1
            n_oxygen = sum(1 for i in inds
                           if mol.GetAtomWithIdx(i).GetSymbol() == "O")
            check(f"{lead} / {core}: oxygens pinned", n_oxygen, want_o)

            acyclic_n = [
                i for i in inds
                if mol.GetAtomWithIdx(i).GetSymbol() == "N"
                and not mol.GetAtomWithIdx(i).IsInRing()
                and mol.GetAtomWithIdx(i).GetDegree() == 3
            ]
            check(f"{lead} / {core}: acyclic P2 nitrogens pinned",
                  len(acyclic_n), 1)

            if acyclic_n:
                free = [nb.GetIdx()
                        for nb in mol.GetAtomWithIdx(acyclic_n[0]).GetNeighbors()
                        if nb.GetIdx() not in inds]
                # both substituents must be free, or they cannot be regenerated
                check(f"{lead} / {core}: free groups on P2 nitrogen", len(free), 2)


# =============================================================================
# 3. N_x1 sizing
# =============================================================================
def test_sizing() -> None:
    section("3. negative --add-atoms-range parses and sizes correctly")

    p = argparse.ArgumentParser()
    p.add_argument("--add-atoms-range", nargs=2, type=int, default=[0, 0])
    try:
        got = p.parse_args(["--add-atoms-range", "-6", "-2"]).add_atoms_range
    except SystemExit:
        got = "ARGPARSE REJECTED"
    check("space-separated negatives parse", got, [-6, -2])

    # The '=' form is a known footgun; confirm it still fails so the comment in
    # the job scripts stays true.
    try:
        p.parse_args(["--add-atoms-range=-6", "-2"])
        eq_ok = True
    except SystemExit:
        eq_ok = False
    check("'=' form rejected (documented footgun)", eq_ok, False)

    for lead, n_ref, lo, hi in (("Cpd12", 74, 68, 72), ("Cpd18", 75, 69, 73)):
        mol = Chem.AddHs(Chem.MolFromSmiles(leads.LEADS[lead]["smiles"]))
        check(f"{lead}: reference N_x1 (atoms incl. H)", mol.GetNumAtoms(), n_ref)
        check(f"{lead}: N_x1 range with offset -6..-2",
              [n_ref - 6, n_ref - 2], [lo, hi])

        mw_per_atom = Descriptors.MolWt(Chem.RemoveHs(mol)) / n_ref
        est_lo, est_hi = lo * mw_per_atom, hi * mw_per_atom
        check(f"{lead}: implied MW band stays under 500",
              f"{est_lo:.0f}-{est_hi:.0f} Da", "< 500", ok=est_hi < 500)


# =============================================================================
# 4. filter verdicts on known molecules
# =============================================================================
def test_filter_logic() -> None:
    section("4. filter verdicts on molecules with known answers")

    cases = [
        # (label, smiles, expect_pass, must_contain_reason, threshold overrides)
        ("Cpd12 (MW 495.6, rotB 10)", leads.LEADS["Cpd12"]["smiles"], True, None, {}),
        ("Cpd18 (MW 507.7 -> over)", leads.LEADS["Cpd18"]["smiles"], False, "MW>500", {}),
        ("Cpd1  (MW 413.5)", leads.LEADS["Cpd1"]["smiles"], True, None, {}),
        ("Cpd12 under a 400 cap", leads.LEADS["Cpd12"]["smiles"], False, "MW>400",
         {"max_mw": 400.0}),
        ("Cpd12 under rotB<=7", leads.LEADS["Cpd12"]["smiles"], False, "rotB>7",
         {"max_rotb": 7}),
        ("14-membered macrocycle", "N#Cc1ncncn1.C1CCCCCCCCCCCCC1", False,
         "ring14>7", {"require_triazine": False}),
        ("no nitrile", "O=C(N1CCOCC1)c1ncncn1", False, "no-nitrile", {}),
    ]

    pains = properties.pains_catalog()
    for label, smi, want_pass, want_reason, over in cases:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            check(f"{label}: parses", False, True)
            continue
        d = properties.describe(mol)
        why = properties.reasons_to_reject(
            mol, d, properties.thresholds(**over), pains
        )
        check(f"{label}: verdict", "PASS" if not why else f"REJECT {why}",
              "PASS" if want_pass else f"REJECT containing {want_reason!r}",
              ok=(not why) if want_pass else (want_reason in why))

    # thresholds() should refuse a typo rather than silently ignoring it
    try:
        properties.thresholds(max_mwt=400)
        typo_caught = False
    except TypeError:
        typo_caught = True
    check("thresholds() rejects an unknown key", typo_caught, True)


# =============================================================================
# 5. filter end-to-end: does it write what downstream docking needs?
# =============================================================================
def test_filter_io() -> None:
    section("5. filter writes passing.smi and passing.sdf")

    # Two molecules that pass at MW<=500, one that fails on MW.
    smis = [leads.LEADS["Cpd12"]["smiles"],
            leads.LEADS["Cpd1"]["smiles"],
            leads.LEADS["Cpd18"]["smiles"]]

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td) / "run" / "ref"
        run_dir.mkdir(parents=True)
        w = Chem.SDWriter(str(run_dir / "generated_molecules.sdf"))
        for s in smis:
            m = Chem.AddHs(Chem.MolFromSmiles(s))
            if AllChem.EmbedMolecule(m, randomSeed=1) != 0:
                continue
            w.write(m)
        w.close()

        proc = subprocess.run(
            [sys.executable, str(GENERATE / "filter_generated.py"), str(Path(td) / "run"),
             "--max-mw", "500", "--max-rotb", "10", "--max-ring-size", "7"],
            capture_output=True, text=True, cwd=str(REPO),
        )
        check("filter exits cleanly", proc.returncode, 0)
        if proc.returncode != 0:
            print("        " + (proc.stderr.strip().splitlines() or ["?"])[-1][:200])

        smi_out = Path(td) / "run" / "passing.smi"
        sdf_out = Path(td) / "run" / "passing.sdf"
        check("passing.smi written", smi_out.exists(), True)
        check("passing.sdf written (docking input)", sdf_out.exists(), True)

        if sdf_out.exists():
            kept = [m for m in Chem.SDMolSupplier(str(sdf_out)) if m is not None]
            check("passing.sdf molecule count (Cpd18 excluded on MW)", len(kept), 2)
            has_3d = all(m.GetNumConformers() > 0 and m.GetConformer().Is3D()
                         for m in kept)
            check("passing.sdf carries 3D conformers", has_3d, True)


# =============================================================================
# 6. optional: a real 2-sample generation
# =============================================================================
def test_generation(device: str) -> None:
    section(f"6. live 2-sample generation on {device} (slow)")

    for core in ("warhead_p2n", "warhead_morpholine_p2n"):
        with tempfile.TemporaryDirectory() as td:
            cmd = [
                sys.executable, str(TGCPL_PKG / "tgcpl_generate.py"),
                "--ref-smiles", "Cpd12", "--n-samples", "2", "--batch-size", "2",
                "--condition", "all", "--scaffold-constraint",
                "--scaffold-core", core,
                "--add-atoms-range", "-6", "-2",
                "--device", device, "--output-dir", td, "--seed", "42",
            ]
            print(f"  running: --scaffold-core {core} ...")
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
            check(f"{core}: generation exits cleanly", proc.returncode, 0)
            if proc.returncode != 0:
                print("        " + (proc.stderr.strip().splitlines() or ["?"])[-1][:200])
                continue
            found = list(Path(td).rglob("generated_molecules.sdf"))
            check(f"{core}: wrote an SDF", bool(found), True)
            if found:
                mols = [m for m in Chem.SDMolSupplier(str(found[0])) if m is not None]
                check(f"{core}: produced >=1 valid molecule", len(mols) >= 1, True)
                for m in mols:
                    print(f"        -> MW {Descriptors.MolWt(Chem.RemoveHs(m)):.1f}, "
                          f"rotB {rdmd.CalcNumRotatableBonds(m)}, "
                          f"{Chem.MolToSmiles(Chem.RemoveHs(m))[:60]}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true",
                    help="also run a real 2-sample generation (slow: minutes on "
                         "CPU, and it loads the model)")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                    help="device for --full. NOTE: mps does not work -- "
                         "torch_cluster has no Metal kernels")
    a = ap.parse_args()

    print("=" * 70)
    print("  Pre-flight: scaffold cores, sizing, property filter")
    print("=" * 70)

    test_cores()
    test_sizing()
    test_filter_logic()
    test_filter_io()
    if a.full:
        test_generation(a.device)

    n_fail = sum(1 for r, _, _ in _results if r == FAIL)
    print("\n" + "=" * 70)
    if n_fail:
        print(f"  {n_fail} of {len(_results)} CHECKS FAILED")
        for r, name, detail in _results:
            if r == FAIL:
                print(f"    FAIL  {name}  ({detail})")
        print("=" * 70)
        sys.exit(1)
    print(f"  ALL CHECKS PASSED ({len(_results)})")
    print("=" * 70)


if __name__ == "__main__":
    main()
