"""ShEPhERD-2 generation of TgCPL-selective covalent inhibitors.

Selectivity comes from the S2 pocket: TgCPL is acidic (Glu75/Asp218), HsCPL
hydrophobic (Leu69/Ala214). Runs condition on a lead's shape/ESP/pharmacophore
profile.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from shepherd import load_model
from shepherd.extract import mol_charges_from_samples
from shepherd.interaction_profile import extract_interaction_profile

torch.set_float32_matmul_precision("high")

# Run as a script from inside the package, so sys.path[0] is scripts/tgcpl/,
# not scripts/. Add the parent so `import tgcpl` resolves from any cwd.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from tgcpl.conformers import embed_lead                       # noqa: E402,F401
from tgcpl.leads import LEADS, lead_names, registry_id        # noqa: E402,F401
from tgcpl.scaffolds import (                                 # noqa: E402,F401
    DEFAULT_CORE,
    SCAFFOLD_CORES,
    TRIAZINE_CORE_SMARTS,
    core_names,
    get_scaffold_atom_indices,
)


def run_generation(
    ref_name: str,
    ref_smiles: str,
    model,
    n_samples: int,
    batch_size: int,
    condition: str,
    add_atoms_min: int,
    add_atoms_max: int,
    add_pharms_min: int,
    add_pharms_max: int,
    neutralize_esp: bool,
    use_scaffold_constraint: bool,
    scaffold_core: str,
    output_dir: Path,
    device: str,
    seed: int,
) -> list[Chem.Mol]:
    """Generate molecules conditioned on one reference lead."""
    ref_dir = output_dir / ref_name
    ref_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Reference: {ref_name}")
    print(f"  SMILES: {ref_smiles}")
    print(f"  Condition: {condition} | n_samples={n_samples}")
    print(f"{'='*60}")

    print("  Embedding reference conformer...")
    ref_mol = embed_lead(ref_smiles, seed=seed)
    if ref_mol is None:
        print(f"  ERROR: Could not embed {ref_name}")
        return []

    ref_sdf = ref_dir / "reference.sdf"
    writer = Chem.SDWriter(str(ref_sdf))
    writer.write(ref_mol)
    writer.close()

    # Must be decided before profile extraction: the atom indices go into the
    # profile.
    scaffold_atom_indices = None
    if use_scaffold_constraint:
        scaffold_atom_indices = get_scaffold_atom_indices(ref_mol, scaffold_core)
        if scaffold_atom_indices:
            print(f"  Scaffold constraint ({scaffold_core}): {len(scaffold_atom_indices)} atoms fixed")
        else:
            print("  WARNING: Scaffold atoms not matched; running unconstrained")

    print("  Extracting interaction profile (xTB)...")
    profile = extract_interaction_profile(
        ref_mol,
        xtb_optimize=True,
        neutralize_esp=neutralize_esp,
        condition_atom_inds=scaffold_atom_indices,
    )
    if profile is None:
        print(f"  ERROR: Could not extract interaction profile for {ref_name}")
        return []

    print(f"  Generating {n_samples} samples (batch_size={batch_size})...")
    t0 = time.perf_counter()

    gen_kwargs = dict(
        batch_size=batch_size,
        N_x1=profile.n_atoms,
        N_x4=profile.n_pharms,
        condition=profile,
        condition_modalities=condition,
    )
    if scaffold_atom_indices:
        gen_kwargs["scaffold_conditioning"] = True

    # Growing means asking for more atoms/pharmacophores than the reference
    # has. generate() takes the counts directly (per-sample lists allowed), so
    # the range is sampled here rather than passed through.
    grow_atoms = add_atoms_min != 0 or add_atoms_max != 0
    grow_pharms = add_pharms_min != 0 or add_pharms_max != 0
    rng = np.random.default_rng(seed)

    all_mols = []
    n_generated = 0
    while n_generated < n_samples:
        this_batch = min(batch_size, n_samples - n_generated)
        gen_kwargs["batch_size"] = this_batch
        if grow_atoms:
            gen_kwargs["N_x1"] = [
                int(profile.n_atoms + rng.integers(add_atoms_min, add_atoms_max + 1))
                for _ in range(this_batch)
            ]
        if grow_pharms:
            gen_kwargs["N_x4"] = [
                int(profile.n_pharms + rng.integers(add_pharms_min, add_pharms_max + 1))
                for _ in range(this_batch)
            ]
        samples = model.generate(**gen_kwargs)
        n_generated += len(samples)

        mol_charges = mol_charges_from_samples(
            samples, xtb_optimize=True, xtb_timeout=180, verbose=False
        )
        batch_mols = [m for m, _ in mol_charges if m is not None]
        all_mols.extend(batch_mols)
        print(f"    Batch done: {len(batch_mols)}/{this_batch} valid")

        pkl_path = ref_dir / f"samples_batch_{n_generated}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(samples, f)

    elapsed = time.perf_counter() - t0
    print(f"  Total time: {elapsed:.1f}s | Valid mols: {len(all_mols)}/{n_samples}")

    if all_mols:
        out_sdf = ref_dir / "generated_molecules.sdf"
        writer = Chem.SDWriter(str(out_sdf))
        for mol in all_mols:
            if mol is not None:
                writer.write(mol)
        writer.close()
        print(f"  Saved: {out_sdf}")

        out_smi = ref_dir / "generated_smiles.smi"
        out_smi.write_text(
            "\n".join(
                Chem.MolToSmiles(Chem.RemoveHs(m)) for m in all_mols if m is not None
            )
        )
        print(f"  Saved: {out_smi}")

    return all_mols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TgCPL-selective molecules with ShEPhERD-2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ref-smiles",
        choices=list(LEADS.keys()) + ["all"],
        default="all",
        help="Which lead(s) to use as ShEPhERD reference(s)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=50,
        help="Number of molecules to generate per reference",
    )
    parser.add_argument(
        "--batch-size", type=int, default=25,
        help="Samples per model forward pass (tune to your GPU memory)",
    )
    parser.add_argument(
        "--condition", nargs="+", default=["all"],
        choices=["all", "shape", "esp", "pharm"],
        help="Conditioning modalities (repeat to combine, e.g. --condition shape esp)",
    )
    parser.add_argument(
        "--add-atoms-range", nargs=2, type=int, default=[0, 0], metavar=("MIN", "MAX"),
        help="Uniformly sample extra atoms in [MIN, MAX] (0 0 = same size as ref)",
    )
    parser.add_argument(
        "--add-pharms-range", nargs=2, type=int, default=[0, 0], metavar=("MIN", "MAX"),
        help="Uniformly sample extra pharmacophores in [MIN, MAX]",
    )
    parser.add_argument(
        "--neutralize-esp", action="store_true",
        help="Neutralize ESP for conditioning (good for exploring different protonation states)",
    )
    parser.add_argument(
        "--scaffold-constraint", action="store_true",
        help="Fix the scaffold core atoms during generation (see --scaffold-core)",
    )
    parser.add_argument(
        "--scaffold-core", choices=list(SCAFFOLD_CORES), default="warhead_morpholine",
        help="Which core --scaffold-constraint fixes: 'warhead' is the nitrile + "
             "triazine only, freeing P1 so S1 can be explored; 'warhead_morpholine' "
             "also fixes the P1 morpholine",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./tgcpl_outputs"),
        help="Output directory",
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default=None,
        help="cpu or cuda (auto-detect by default)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: CPU inference is very slow. GPU strongly recommended.")

    if args.ref_smiles == "all":
        refs_to_run = list(LEADS.keys())
    else:
        refs_to_run = [args.ref_smiles]

    condition_str = args.condition[0] if len(args.condition) == 1 else "+".join(args.condition)

    print(f"\nTgCPL generation run:")
    print(f"  References:  {refs_to_run}")
    print(f"  Conditioning: {condition_str}")
    print(f"  Samples/ref: {args.n_samples}")
    print(f"  Output:      {args.output_dir}")
    print(f"  Scaffold constraint: {args.scaffold_constraint}"
          + (f" ({args.scaffold_core})" if args.scaffold_constraint else ""))
    print(f"  Extra atoms range: {args.add_atoms_range}")

    # Downloads weights from HuggingFace on first run.
    print("\nLoading ShEPhERD-2 model...")
    model = load_model(device=device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for ref_name in refs_to_run:
        lead = LEADS[ref_name]
        mols = run_generation(
            ref_name=ref_name,
            ref_smiles=lead["smiles"],
            model=model,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            condition=condition_str,
            add_atoms_min=args.add_atoms_range[0],
            add_atoms_max=args.add_atoms_range[1],
            add_pharms_min=args.add_pharms_range[0],
            add_pharms_max=args.add_pharms_range[1],
            neutralize_esp=args.neutralize_esp,
            use_scaffold_constraint=args.scaffold_constraint,
            scaffold_core=args.scaffold_core,
            output_dir=args.output_dir,
            device=device,
            seed=args.seed,
        )
        all_results[ref_name] = mols


    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for ref_name, mols in all_results.items():
        lead = LEADS[ref_name]
        print(
            f"  {ref_name}: {len(mols)} molecules generated "
            f"(ref: IC50={lead['tgcpl_ic50_nM']} nM, sel={lead['selectivity']}x)"
        )
    total = sum(len(v) for v in all_results.values())
    print(f"\n  Total valid molecules: {total}")
    print(f"  Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
