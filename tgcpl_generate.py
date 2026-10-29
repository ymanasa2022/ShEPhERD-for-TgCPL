"""
tgcpl_generate.py
=================
ShEPhERD-2 generation script tailored for TgCPL-selective covalent inhibitors.

Design rationale (from TgCPL Main Draft 6):
--------------------------------------------
TgCPL vs. HsCPL differ in 4 active-site residues within 4.5 Å of the substrate:
  TgCPL:  Gln69,  Glu75,  Asp78,  Asp218   (acidic/polar S2 pocket)
  HsCPL:  Leu69,  Tyr72,  Glu63,  Ala214   (hydrophobic S2 pocket)

Key design goals:
  1. POTENCY    : maintain covalent bond to Cys31; P2 basic amine contacts
                  Glu75 / Asp218 in the acidic TgCPL S2 pocket
  2. SELECTIVITY: large (6-7 membered) cycloalkyl at benzylic P3 carbon
                  blocks 4-fluorobenzyl from fitting HsCPL's hydrophobic S2
  3. NOVEL IDEAS: explore bioisosteres of morpholine (P1), replace 4-F-benzyl
                  with other S3-fitting aryl groups, vary the P2 basic chain

ShEPhERD strategy:
  - Use Cpd12 (629 nM / 37x selective) and Cpd18 (210 nM / 39x selective)
    as primary interaction-profile references
  - Generate molecules conditioned on their SHAPE + ESP + PHARMACOPHORES
  - Optionally neutralise ESP to explore bioisosteres across different charge states
  - Use --add-atoms to explore larger P3 / extended P2 derivatives
  - Scaffold constraint: fix the triazine-morpholine-CN warhead core atoms
  - DUAL-TARGET selectivity run: condition ON TgCPL profile AND NOT HsCPL profile
    using generate_composition() with weights [+0.5, -0.5]. This directly encodes
    "look like Cpd18 in TgCPL, avoid looking like Cpd1 in HsCPL."

Usage:
  conda activate shepherd2
  python tgcpl_generate.py                           # default: all references
  python tgcpl_generate.py --ref-smiles Cpd12        # single reference
  python tgcpl_generate.py --n-samples 100           # more samples
  python tgcpl_generate.py --condition shape esp      # shape+ESP only (no pharm)
  python tgcpl_generate.py --add-atoms 5 10          # grow molecule by 5-10 atoms
  python tgcpl_generate.py --dual-target             # TgCPL AND NOT HsCPL composition
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from shepherd import load_model
from shepherd.comp_inference.utils import align_two_ref_mol_from_mol_charge
from shepherd.extract import mol_charges_from_samples
from shepherd.interaction_profile import extract_interaction_profile, ConditionAtoms
from shepherd_score.conformer_generation import embed_conformer_from_smiles

torch.set_float32_matmul_precision("high")

# ---------------------------------------------------------------------------
# Lead molecules from TgCPL Draft 6
# ---------------------------------------------------------------------------
LEADS = {
    "Cpd1": {
        "smiles": "CCN(CC)CCN(Cc1ccc(F)cc1)c1nc(C#N)nc(N2CCOCC2)n1",
        "tgcpl_ic50_nM": 320,
        "hscpl_ic50_nM": 2240,
        "selectivity": 7.0,
        "note": "Parent compound; no benzylic substituent",
    },
    "Cpd12": {
        "smiles": "CCN(CC)CCN(c1nc(C#N)nc(N2CCOCC2)n1)C(c1ccc(F)cc1)C1CCCCC1",
        "tgcpl_ic50_nM": 629,
        "hscpl_ic50_nM": 23245,
        "selectivity": 36.9,
        "note": "Cyclohexyl benzylic; best single-SAR compound",
    },
    "Cpd13": {
        "smiles": "CCN(CC)CCN(c1nc(C#N)nc(N2CCOCC2)n1)C(c1ccc(F)cc1)C1CCCCCC1",
        "tgcpl_ic50_nM": 497,
        "hscpl_ic50_nM": 19695,
        "selectivity": 39.6,
        "note": "Cycloheptyl benzylic; best selectivity overall",
    },
    "Cpd18": {
        "smiles": "N#Cc1nc(N2CCOCC2)nc(N(CCC2CCNCC2)C(c2ccc(F)cc2)C2CCCCC2)n1",
        "tgcpl_ic50_nM": 210,
        "hscpl_ic50_nM": 8080,
        "selectivity": 38.5,
        "note": "Best potency + high selectivity; piperidinylethyl P2",
    },
}

# Scaffold cores for --scaffold-constraint, smallest first.
#
# "warhead" is just the covalent nitrile and the triazine it hangs off: the part
# that is mechanistically required, since the C#N is what reacts with the
# catalytic cysteine. Everything else — including the morpholine at P1 — is left
# for ShEPhERD to rebuild, so S1 is explorable.
#
# "warhead_morpholine" additionally fixes the P1 morpholine. On Cpd18 that is 14
# of 37 heavy atoms versus 8, so it leaves markedly less room to generate into.
#
# Caveat for "warhead": the nitrile's electrophilicity depends on how electron
# -poor the triazine is, which the P1 substituent tunes. Replacing morpholine
# changes covalent reactivity, so generated analogues need a reactivity check,
# not just a shape score.
SCAFFOLD_CORES = {
    "warhead": "N#Cc1ncncn1",
    "warhead_morpholine": "N#Cc1nc(N2CCOCC2)ncn1",
}

# Back-compat alias; previously the only core.
TRIAZINE_CORE_SMARTS = SCAFFOLD_CORES["warhead_morpholine"]


def embed_lead(smiles: str, seed: int = 42) -> Chem.Mol | None:
    """Embed lead SMILES as an xTB-relaxed 3D conformer with explicit H."""
    mol = embed_conformer_from_smiles(smiles, MMFF_optimize=True, random_seed=seed)
    return mol


def get_scaffold_atom_indices(mol: Chem.Mol, core: str = "warhead_morpholine") -> list[int]:
    """Return atom indices matching the chosen scaffold core (see SCAFFOLD_CORES)."""
    core_query = Chem.MolFromSmarts(SCAFFOLD_CORES[core])
    if core_query is None:
        return []
    matches = mol.GetSubstructMatches(core_query, useChirality=False)
    if not matches:
        return []
    # Flatten all matches (take first if multiple)
    return list(matches[0])


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

    # Embed reference
    print("  Embedding reference conformer...")
    ref_mol = embed_lead(ref_smiles, seed=seed)
    if ref_mol is None:
        print(f"  ERROR: Could not embed {ref_name}")
        return []

    # Save reference SDF
    ref_sdf = ref_dir / "reference.sdf"
    writer = Chem.SDWriter(str(ref_sdf))
    writer.write(ref_mol)
    writer.close()

    # Scaffold constraint (optional): fix the core atoms. This has to be decided
    # before profile extraction, because the atom indices go into the profile.
    scaffold_atom_indices = None
    if use_scaffold_constraint:
        scaffold_atom_indices = get_scaffold_atom_indices(ref_mol, scaffold_core)
        if scaffold_atom_indices:
            print(f"  Scaffold constraint ({scaffold_core}): {len(scaffold_atom_indices)} atoms fixed")
        else:
            print("  WARNING: Scaffold atoms not matched; running unconstrained")

    # Extract interaction profile (shape + ESP + pharmacophores)
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

    # Generate
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

    # "Grow" the molecule by asking for more atoms/pharmacophores than the
    # reference has. generate() takes the counts directly (per-sample lists are
    # allowed), so the range is sampled here rather than passed through.
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

        # Convert to RDKit mols via xTB relaxation
        mol_charges = mol_charges_from_samples(
            samples, xtb_optimize=True, xtb_timeout=180, verbose=False
        )
        batch_mols = [m for m, _ in mol_charges if m is not None]
        all_mols.extend(batch_mols)
        print(f"    Batch done: {len(batch_mols)}/{this_batch} valid")

        # Save batch samples (raw)
        pkl_path = ref_dir / f"samples_batch_{n_generated}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(samples, f)

    elapsed = time.perf_counter() - t0
    print(f"  Total time: {elapsed:.1f}s | Valid mols: {len(all_mols)}/{n_samples}")

    # Save all valid molecules
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


def run_dual_target_selectivity(
    tgcpl_ref_name: str,
    tgcpl_smiles: str,
    hscpl_ref_name: str,
    hscpl_smiles: str,
    model,
    n_samples: int,
    batch_size: int,
    use_scaffold_constraint: bool,
    scaffold_core: str,
    output_dir: Path,
    seed: int,
) -> list[Chem.Mol]:
    """
    Dual-target selectivity run using ShEPhERD-2 compositional generation.

    Strategy: condition ON the TgCPL lead's interaction profile AND NOT the HsCPL
    reference's profile. ShEPhERD uses weights [+0.5, -0.5] with
    composition_mode='conditional', which steers generation toward TgCPL-matching
    molecules that simultaneously avoid the HsCPL electrostatic/shape profile.

    This directly exploits:
      - TgCPL's acidic S2 (Glu75, Asp218) → encoded in the positive TgCPL ESP profile
      - HsCPL's hydrophobic S2 (Leu69, Ala214) → negated by the negative HsCPL weight
      - The deeper, narrower TgCPL S2 → encoded in the positive shape profile

    The two references are aligned by ESP similarity before profile extraction, so
    they share a common 3D reference frame.

    Parameters
    ----------
    tgcpl_ref_name : str
        Label for the TgCPL reference (positive target), e.g. "Cpd18"
    tgcpl_smiles : str
        SMILES of the TgCPL-selective lead (e.g. Cpd18: best potency + selectivity)
    hscpl_ref_name : str
        Label for the HsCPL reference (anti-target), e.g. "Cpd1"
    hscpl_smiles : str
        SMILES of a compound with good HsCPL activity (e.g. Cpd1: selectivity 7x,
        high HsCPL engagement) — provides the HsCPL interaction profile to avoid
    model : ShEPhERD model
    n_samples : int
    batch_size : int
    use_scaffold_constraint : bool
        If True, fixes the triazine-CN-morpholine warhead from the TgCPL reference
    output_dir : Path
    seed : int
    """
    run_label = f"dual_{tgcpl_ref_name}_AND_NOT_{hscpl_ref_name}"
    ref_dir = output_dir / run_label
    ref_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  DUAL-TARGET SELECTIVITY RUN")
    print(f"  Positive target (TgCPL): {tgcpl_ref_name}  weight=+0.5")
    print(f"  Anti-target  (HsCPL):    {hscpl_ref_name}  weight=-0.5")
    print(f"  n_samples={n_samples}")
    print(f"{'='*60}")

    # --- 1. Embed both references ---
    print("  Embedding TgCPL reference...")
    tgcpl_mol = embed_lead(tgcpl_smiles, seed=seed)
    if tgcpl_mol is None:
        print(f"  ERROR: Could not embed TgCPL reference ({tgcpl_ref_name})")
        return []

    print("  Embedding HsCPL reference...")
    hscpl_mol = embed_lead(hscpl_smiles, seed=seed)
    if hscpl_mol is None:
        print(f"  ERROR: Could not embed HsCPL reference ({hscpl_ref_name})")
        return []

    # --- 2. Align hscpl_mol onto tgcpl_mol by ESP similarity ---
    # This puts both molecules in the same 3D frame, required for composition.
    # We align HsCPL onto TgCPL so the TgCPL molecule is the anchor frame.
    print("  Aligning HsCPL reference onto TgCPL frame (ESP alignment)...")
    _, hscpl_mol_aligned = align_two_ref_mol_from_mol_charge(
        mol_to_align_mol=hscpl_mol,
        ref_mol=tgcpl_mol,
        condition="esp",
        do_center=True,
    )
    print("  Alignment done.")

    # Save reference SDFs for inspection
    for mol, name in [(tgcpl_mol, "tgcpl_ref.sdf"), (hscpl_mol_aligned, "hscpl_ref_aligned.sdf")]:
        writer = Chem.SDWriter(str(ref_dir / name))
        writer.write(mol)
        writer.close()

    # --- 3. Extract interaction profiles ---
    print("  Extracting TgCPL interaction profile (xTB)...")
    profile_tgcpl = extract_interaction_profile(tgcpl_mol, xtb_optimize=True, center=True)
    if profile_tgcpl is None:
        print("  ERROR: Could not extract TgCPL interaction profile")
        return []

    print("  Extracting HsCPL interaction profile (xTB)...")
    profile_hscpl = extract_interaction_profile(hscpl_mol_aligned, xtb_optimize=True, center=True)
    if profile_hscpl is None:
        print("  ERROR: Could not extract HsCPL interaction profile")
        return []

    # Re-translate profile_hscpl into TgCPL frame (same COM reference)
    profile_hscpl.translated(
        profile_tgcpl.com_before_centering - profile_hscpl.com_before_centering
    )

    print(f"  TgCPL profile: n_atoms={profile_tgcpl.n_atoms}, n_pharms={profile_tgcpl.n_pharms}")
    print(f"  HsCPL profile: n_atoms={profile_hscpl.n_atoms}, n_pharms={profile_hscpl.n_pharms}")

    # --- 4. Scaffold constraint (optional) ---
    scaffold_condition = None
    if use_scaffold_constraint:
        core_atom_inds = get_scaffold_atom_indices(tgcpl_mol, scaffold_core)
        if core_atom_inds:
            scaffold_condition = ConditionAtoms.from_mol(tgcpl_mol, inds=core_atom_inds)
            print(f"  Scaffold constraint ({scaffold_core}): {len(core_atom_inds)} core atoms fixed")
        else:
            print("  WARNING: Scaffold not matched; running without constraint")

    # --- 5. Compositional generation ---
    # weights = [+0.5, -0.5]:  satisfy TgCPL AND NOT HsCPL
    # composition_mode='conditional': best for AND NOT (from ShEPhERD-2 tutorial)
    weights = np.array([0.5, -0.5])
    N_edm = 400  # default EDM steps
    early_stop = int(0.9 * N_edm)

    print(f"  Running generate_composition (weights={weights}, "
          f"n_samples={n_samples}, batch_size={batch_size})...")
    t0 = time.perf_counter()

    all_samples = []
    n_generated = 0
    while n_generated < n_samples:
        this_batch = min(batch_size, n_samples - n_generated)
        comp_kwargs = dict(
            N_x1=profile_tgcpl.n_atoms,
            N_x4=profile_tgcpl.n_pharms,
            batch_size=this_batch,
            profiles=[profile_tgcpl, profile_hscpl],
            weights_conditions=weights,
            composition_mode="conditional",
            early_stop_edm=early_stop,
        )
        if scaffold_condition is not None:
            comp_kwargs["atom_condition"] = scaffold_condition
            comp_kwargs["scaffold_conditioning"] = True

        samples = model.generate_composition(**comp_kwargs)
        all_samples.extend(samples)
        n_generated += len(samples)

        # Save raw batch
        pkl_path = ref_dir / f"samples_batch_{n_generated}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(samples, f)
        print(f"    Batch done: {len(samples)} samples")

    # --- 6. Convert to RDKit molecules ---
    print("  Converting samples to RDKit molecules (xTB relaxation)...")
    mol_charges = mol_charges_from_samples(
        all_samples, xtb_optimize=True, xtb_timeout=180, verbose=False
    )
    all_mols = [m for m, _ in mol_charges if m is not None]
    elapsed = time.perf_counter() - t0

    print(f"  Total time: {elapsed:.1f}s | Valid mols: {len(all_mols)}/{n_samples}")

    # --- 7. Save outputs ---
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

        # Write run summary
        (ref_dir / "run_info.txt").write_text(
            f"Dual-target selectivity run\n"
            f"Positive target: {tgcpl_ref_name} (TgCPL, weight=+0.5)\n"
            f"Anti-target:     {hscpl_ref_name} (HsCPL, weight=-0.5)\n"
            f"Composition mode: conditional\n"
            f"Scaffold constraint: {use_scaffold_constraint}\n"
            f"n_samples requested: {n_samples} | valid: {len(all_mols)}\n"
            f"Time: {elapsed:.1f}s\n"
        )

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
    parser.add_argument(
        "--dual-target", action="store_true",
        help=(
            "Run dual-target selectivity generation using generate_composition(). "
            "Conditions ON Cpd18 (TgCPL, weight=+0.5) AND NOT Cpd1 (HsCPL, weight=-0.5). "
            "Directly encodes selectivity at the model level. "
            "Can be combined with --scaffold-constraint."
        ),
    )
    parser.add_argument(
        "--dual-tgcpl-ref", default="Cpd18",
        choices=list(LEADS.keys()),
        help="TgCPL reference lead (positive target) for dual-target run",
    )
    parser.add_argument(
        "--dual-hscpl-ref", default="Cpd1",
        choices=list(LEADS.keys()),
        help="HsCPL anti-target reference for dual-target run. "
             "Cpd1 is a good default: it has good HsCPL activity (IC50=2240 nM) "
             "and minimal benzylic bulk, so its HsCPL profile captures hydrophobic S2 engagement.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: CPU inference is very slow. GPU strongly recommended.")

    # Determine which leads to run
    if args.ref_smiles == "all":
        refs_to_run = list(LEADS.keys())
    else:
        refs_to_run = [args.ref_smiles]

    # Condition string: "all" or "shape+esp" etc.
    condition_str = args.condition[0] if len(args.condition) == 1 else "+".join(args.condition)

    print(f"\nTgCPL generation run:")
    print(f"  References:  {refs_to_run}")
    print(f"  Conditioning: {condition_str}")
    print(f"  Samples/ref: {args.n_samples}")
    print(f"  Output:      {args.output_dir}")
    print(f"  Scaffold constraint: {args.scaffold_constraint}"
          + (f" ({args.scaffold_core})" if args.scaffold_constraint else ""))
    print(f"  Extra atoms range: {args.add_atoms_range}")

    # Load model (downloads weights from HuggingFace on first run)
    print("\nLoading ShEPhERD-2 model...")
    model = load_model(device=device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # --- Standard single-reference runs ---
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

    # --- Dual-target selectivity run (TgCPL AND NOT HsCPL) ---
    if args.dual_target:
        tgcpl_lead = LEADS[args.dual_tgcpl_ref]
        hscpl_lead = LEADS[args.dual_hscpl_ref]
        dual_mols = run_dual_target_selectivity(
            tgcpl_ref_name=args.dual_tgcpl_ref,
            tgcpl_smiles=tgcpl_lead["smiles"],
            hscpl_ref_name=args.dual_hscpl_ref,
            hscpl_smiles=hscpl_lead["smiles"],
            model=model,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            use_scaffold_constraint=args.scaffold_constraint,
            scaffold_core=args.scaffold_core,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        all_results[f"dual_{args.dual_tgcpl_ref}_AND_NOT_{args.dual_hscpl_ref}"] = dual_mols

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for ref_name, mols in all_results.items():
        if ref_name.startswith("dual_"):
            print(f"  {ref_name}: {len(mols)} molecules generated [DUAL-TARGET]")
        else:
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
