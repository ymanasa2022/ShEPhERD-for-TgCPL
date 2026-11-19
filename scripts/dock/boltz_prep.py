"""
Turn generated molecules into Boltz-2 inputs for BOTH targets, so each ligand
is co-folded against TgCPL (3F75) and HsCPL (5MAJ). Boltz-2 takes one affinity
binder per input, hence one YAML per ligand per target; the shared ligand id is
the join key boltz_selectivity.py pairs the two runs on.

The four leads are written in as CTRL_<name> by default: they have measured
IC50s against both enzymes, so they say whether the predicted ordering
reproduces the experimental one before any novel molecule is believed.

affinity_pred_value is log10(IC50 in uM), so LOWER is stronger and selectivity
is hscpl - tgcpl. The affinity head was trained to rank actives against ONE
target, so differencing two independent predictions across two homologues is
off-label -- the controls are what say whether it carries signal here. Nothing
here models the covalent bond to Cys31/Cys25.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tgcpl lives under scripts/

from tgcpl.leads import LEADS  # noqa: E402

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET_DIR = REPO_ROOT / "data" / "targets"

#: dir name -> fasta basename; fetch_targets.sh regenerates these from RCSB.
TARGETS = {
    "tgcpl": "tgcpl_3f75.fasta",
    "hscpl": "hscpl_5maj.fasta",
}

#: Boltz-2 rejects an affinity ligand above 128 atoms and advises under 56.
BOLTZ_ATOM_LIMIT = 128
BOLTZ_ATOM_ADVISED = 56

PROTEIN_CHAIN = "A"
LIGAND_CHAIN = "L"


def read_fasta(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(
            f"missing target sequence: {path}\n"
            f"Run: bash scripts/dock/fetch_targets.sh"
        )
    seq = "".join(
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith(">")
    )
    if not seq:
        raise SystemExit(f"no sequence in {path}")
    return seq


def load_ligands(src: Path) -> list[tuple[str, str]]:
    """(smiles, label) from .sdf, .smi or selectivity.tsv, in file order."""
    out: list[tuple[str, str]] = []
    if src.suffix.lower() == ".sdf":
        for mol in Chem.SDMolSupplier(str(src), removeHs=True):
            if mol is None:
                continue
            label = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            out.append((Chem.MolToSmiles(mol), label))
        return out

    for line in src.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t") if "\t" in line else line.split()
        smi = fields[0]  # skips selectivity.tsv's header: not a SMILES
        if Chem.MolFromSmiles(smi) is None:
            continue
        out.append((smi, fields[1] if len(fields) > 1 else ""))
    return out


def boltz_atom_count(smi: str) -> int | None:
    """Atoms Boltz counts: heavy plus the hydrogens RemoveHs keeps."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.RemoveHs(Chem.AddHs(mol)).GetNumAtoms()


def write_yaml(path: Path, sequence: str, smiles: str,
               msa: Path | None = None) -> None:
    """One Boltz-2 input: the target, one ligand, affinity on that ligand.

    Without an msa: field the run needs --use_msa_server; a cached a3m makes
    array tasks and re-runs independent of it.

    The SMILES MUST stay quoted: these contain '#' for the nitrile, which
    unquoted starts a YAML comment and silently truncates every ligand at the
    warhead.
    """
    msa_line = f"      msa: {msa}\n" if msa is not None else ""
    path.write_text(
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        f"      id: {PROTEIN_CHAIN}\n"
        f"      sequence: {sequence}\n"
        f"{msa_line}"
        "  - ligand:\n"
        f"      id: {LIGAND_CHAIN}\n"
        f"      smiles: '{smiles}'\n"
        "properties:\n"
        "  - affinity:\n"
        f"      binder: {LIGAND_CHAIN}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("ligands", type=Path,
                    help="selected.sdf, passing.smi, or selectivity.tsv")
    ap.add_argument("-o", "--out-dir", type=Path, default=Path("boltz_in"))
    ap.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    ap.add_argument("--msa-dir", type=Path, default=None,
                    help="Directory holding tgcpl.a3m / hscpl.a3m. Given, the "
                         "YAMLs point at them and the run needs no MSA server")
    ap.add_argument("--no-controls", dest="controls", action="store_false",
                    help="Skip the four leads. Only when you already have them")
    ap.add_argument("--limit", type=int, default=0,
                    help="Prepare only the first N generated ligands")
    args = ap.parse_args()

    seqs = {t: read_fasta(args.target_dir / f) for t, f in TARGETS.items()}
    for t, s in seqs.items():
        print(f"{t:<6} {len(s)} aa  {args.target_dir / TARGETS[t]}")

    ligands = load_ligands(args.ligands)
    if not ligands:
        raise SystemExit(f"no ligands parsed from {args.ligands}")
    if args.limit:
        ligands = ligands[: args.limit]
    print(f"\nLigands from {args.ligands}: {len(ligands)}")

    # Deduped: a repeated SMILES wastes a GPU-hour per target and breaks the join.
    entries: dict[str, tuple[str, str]] = {}
    seen_smiles: dict[str, str] = {}
    n_dup = 0
    for i, (smi, label) in enumerate(ligands, 1):
        if smi in seen_smiles:
            n_dup += 1
            continue
        lid = f"L{i:04d}"
        seen_smiles[smi] = lid
        entries[lid] = (smi, label)
    if n_dup:
        print(f"  (dropped {n_dup} duplicate SMILES)")

    if args.controls:
        for name, lead in LEADS.items():
            smi = Chem.MolToSmiles(Chem.MolFromSmiles(lead["smiles"]))
            lid = f"CTRL_{name}"
            if smi in seen_smiles:
                print(f"  note: {name} is also in the generated set "
                      f"({seen_smiles[smi]}); keeping both ids")
            entries[lid] = (smi, f"control:{name}")
        print(f"  + {len(LEADS)} controls")

    too_big, advisory = [], []
    for lid, (smi, _) in list(entries.items()):
        n = boltz_atom_count(smi)
        if n is None:
            del entries[lid]
            continue
        if n > BOLTZ_ATOM_LIMIT:
            too_big.append((lid, n))
            del entries[lid]
        elif n > BOLTZ_ATOM_ADVISED:
            advisory.append((lid, n))
    if too_big:
        print(f"\n  DROPPED {len(too_big)} over the {BOLTZ_ATOM_LIMIT}-atom "
              f"affinity limit: {', '.join(l for l, _ in too_big[:8])}"
              + (" ..." if len(too_big) > 8 else ""))
    if advisory:
        n_max = max(n for _, n in advisory)
        print(f"  {len(advisory)} ligands exceed the advised {BOLTZ_ATOM_ADVISED} "
              f"atoms (largest {n_max}). Allowed, but the affinity head is "
              f"out of its comfortable range -- this chemotype simply is.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for target, seq in seqs.items():
        tdir = args.out_dir / target
        tdir.mkdir(exist_ok=True)
        msa = None
        if args.msa_dir is not None:
            candidate = (args.msa_dir / f"{target}.a3m").resolve()
            if candidate.is_file():
                msa = candidate
                print(f"\n{target}: using cached MSA {msa}")
            else:
                print(f"\n{target}: no {candidate}, falling back to "
                      f"--use_msa_server")
        for lid, (smi, _) in entries.items():
            write_yaml(tdir / f"{lid}.yaml", seq, smi, msa=msa)
        print(f"Wrote {len(entries)} YAML -> {tdir}")

    table = args.out_dir / "ligands.tsv"
    with table.open("w") as fh:
        fh.write("# written by boltz_prep.py\n")
        fh.write("ligand_id\tsmiles\tsource_label\tis_control\n")
        for lid, (smi, label) in entries.items():
            fh.write(f"{lid}\t{smi}\t{label or 'NA'}\t"
                     f"{1 if lid.startswith('CTRL_') else 0}\n")
    print(f"Wrote {table}")

    n = len(entries)
    print(f"\n{n} ligands x 2 targets = {2 * n} Boltz-2 predictions.")
    print("\nNext:")
    print(f"  boltz predict {args.out_dir}/tgcpl --use_msa_server "
          f"--out_dir boltz_out/tgcpl")
    print(f"  boltz predict {args.out_dir}/hscpl --use_msa_server "
          f"--out_dir boltz_out/hscpl")
    print("  python scripts/dock/boltz_selectivity.py boltz_out "
          f"--ligands {table}")
    print("\nOn Great Lakes: sbatch slurm_scripts/greatlakes_boltz_selectivity.sbatch")


if __name__ == "__main__":
    main()
