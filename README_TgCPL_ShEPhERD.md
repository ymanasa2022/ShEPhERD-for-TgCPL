# TgCPL ShEPhERD-2 — Setup & Drug Design Guide

## What's in this folder

| File | Purpose |
|------|---------|
| `setup_shepherd2.sh` | One-shot conda/pip install for HPC |
| `tgcpl_generate.py` | Main generation script with TgCPL-aware leads/constraints |
| `slurm_tgcpl.sh` | SLURM batch job (4 designed runs) |
| `lead_molecules.smi` | SMILES + IC50/selectivity for all leads from Draft 6 |

---

## Quick start (HPC)

```bash
# 1. Upload these files to your HPC home directory alongside the cloned repo
scp -r shepherd2_setup/ hpc_username@hpc.host.edu:~/

# 2. Clone + install (choose your CUDA version: 118/124/126/128/130)
bash setup_shepherd2.sh 126

# 3. Activate and run a quick smoke test
conda activate shepherd2
cd ~/shepherd2
python examples/basic_inference/basic_inference_test.py \
    --smiles "CCN(CC)CCN(c1nc(C#N)nc(N2CCOCC2)n1)C(c1ccc(F)cc1)C1CCCCC1" \
    --output-dir test_output \
    --batch-size 5

# 4. Run TgCPL-targeted generation (SLURM)
mkdir -p logs
sbatch slurm_tgcpl.sh
```

---

## Scientific design rationale

### TgCPL vs HsCPL active site differences

Only **4 residues** differ within 4.5 Å of the binding site:

| Sub-pocket | TgCPL residue | HsCPL residue | Impact |
|------------|--------------|--------------|--------|
| S2 | **Gln69** | Leu69 | TgCPL is polar; HsCPL hydrophobic |
| S2 | **Glu75** | Tyr72 | TgCPL acidic; important for P2 amine contacts |
| S2 | **Asp78** | — | Additional acidic lining (TgCPL-specific) |
| S2 | **Asp218** | Ala214 | TgCPL acidic; HsCPL non-polar |
| S2 | — | Glu63 | HsCPL-specific; forms halogen bond with bulky P3 |

**Key structural insight**: TgCPL's S2 pocket is **narrower and predominantly acidic (negatively charged)**, while HsCPL's S2 is **larger and hydrophobic**. This creates the selectivity handle.

### How lead compounds achieve selectivity

```
Compound 1 (no benzylic sub):  
  → 4-fluorobenzyl is FREE to rotate → occupies deep S2 pocket  
  → Works in BOTH TgCPL (via Gln69 H-bond) AND HsCPL (via hydrophobic contacts)  
  → Selectivity = 7x (not great)

Compound 12 / 13 (cyclohexyl / cycloheptyl benzylic):  
  → Bulky cycloalkyl LOCKS the benzylic carbon → forces 4-fluorobenzyl into S3  
  → P2 diethylamine now occupies S2 → interacts with TgCPL's Glu75 + Asp218  
  → These acidic residues are ABSENT in HsCPL → HsCPL loses affinity  
  → TgCPL IC50 ~600 nM, Selectivity 37-40x ✓

Compound 18 (cyclohexyl P3 + piperidinylethyl P2):  
  → Same P3 mechanism PLUS piperidine NH engages Glu75/Asp218 more effectively  
  → TgCPL IC50 = 210 nM, Selectivity = 38.5x ✓✓ (BEST LEAD)
```

### Goals for ShEPhERD generation

#### Goal 1: Maintain/improve potency (<200 nM TgCPL)
- Keep the covalent nitrile warhead (Cys31 bond essential)
- P2 basic amine must reach Glu75 / Asp218 in S2
- Explore: longer P2 chains, different basic amines (pyrrolidine, piperazine, azepane), piperidine NH vs. NMe

#### Goal 2: Improve selectivity (>50x)
- Maintain large cycloalkyl at benzylic P3 position (cyclohexyl/cycloheptyl optimal so far)
- Explore: bicyclic P3 groups (norbornyl, bicyclo[2.2.2]octyl) that fit TgCPL's narrower S3
- Avoid: gem-disubstitution at benzylic C (decreases both); polar heteroatoms at benzylic C (THP killed potency)
- R-configuration at benzylic carbon preferred (2x better than S)

#### Goal 3: Bioisosteric exploration
- Replace 4-fluorobenzyl (P3 aryl) with other aryl/heteroaryl groups
- Replace morpholine (P1) with other saturated N-heterocycles
- Use `--neutralize-esp` to find bioisosteres across different charge states
- Use `--condition shape esp` (no pharmacophore) for more structurally diverse output

---

## ShEPhERD run strategy

### Run 1: Baseline — all 4 leads, full conditioning
```bash
python tgcpl_generate.py --ref-smiles all --n-samples 50 --condition all
```
Generates 200 molecules conditioned on shape + ESP + pharmacophores of each lead.
Best for finding close analogues with improved drug-like properties.

### Run 2: Shape + ESP from Cpd18 (most potent lead)
```bash
python tgcpl_generate.py --ref-smiles Cpd18 --n-samples 100 --condition shape esp
```
Removes pharmacophore constraint → more structural diversity while preserving
the 3D electrostatic profile matching Glu75/Asp218 interactions.

### Run 3: Grow from Cpd18 (+2 to +6 heavy atoms)
```bash
python tgcpl_generate.py --ref-smiles Cpd18 --n-samples 100 --add-atoms-range 2 6
```
Generates slightly larger molecules. Goal: explore extended P2 chains or
additional substituents that engage TgCPL-specific residues.

### Run 4: Scaffold-constrained from Cpd12
```bash
python tgcpl_generate.py --ref-smiles Cpd12 --n-samples 100 --scaffold-constraint
```
Fixes the triazine-CN-morpholine warhead atoms. Only varies the P2+P3 portions.
Guarantees the covalent warhead is preserved.

### Run 5: Dual-target selectivity — TgCPL AND NOT HsCPL ⭐
```bash
python tgcpl_generate.py --dual-target --dual-tgcpl-ref Cpd18 --dual-hscpl-ref Cpd1 \
    --n-samples 100 --scaffold-constraint
```
Uses ShEPhERD's `generate_composition()` with a signed composition:

```
weights_conditions = [+0.5, -0.5]
profiles = [profile_TgCPL_Cpd18, profile_HsCPL_Cpd1]
composition_mode = 'conditional'
```

This directly encodes **"generate molecules that look like Cpd18 in TgCPL AND do NOT look like Cpd1 in HsCPL"**. The −0.5 weight on the HsCPL profile repels generation away from the hydrophobic HsCPL S2 pocket features, while +0.5 on the TgCPL profile attracts toward the acidic TgCPL S2 pocket features.

**Why Cpd1 as the HsCPL reference?** Cpd1 (IC50 = 2240 nM HsCPL, selectivity = 7x) represents the baseline interaction with HsCPL before selectivity optimization. Its profile captures the undesirable hydrophobic HsCPL S2 contacts we want to avoid.

**Mechanistic interpretation**:
- TgCPL profile (Cpd18): encodes deep acidic S2 pocket; Glu75 + Asp218 contacts; narrow S2 geometry
- HsCPL profile (Cpd1): encodes shallow hydrophobic S2; Leu69 + Ala214 contacts; wider S2 geometry
- The AND NOT composition steers the diffusion denoising process toward TgCPL-fitting electrostatics and away from HsCPL-fitting hydrophobics

**Expected output**: molecules with novel scaffolds that maintain the triazine-nitrile warhead (scaffold constraint) but find new ways to project a basic amine toward TgCPL's acidic residues while avoiding bulky hydrophobic P2 groups that would fit HsCPL.

---

## Post-generation analysis

After generation, score molecules with:

```python
# Quick filtering — run in python or notebook
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

# Load generated molecules
supplier = Chem.SDMolSupplier("tgcpl_outputs/run1_all_leads/Cpd18/generated_molecules.sdf")
mols = [m for m in supplier if m is not None]

# Drug-likeness filter
def passes_filter(mol):
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rotb = Lipinski.NumRotatableBonds(mol)
    return mw < 600 and logp < 5 and hbd <= 5 and hba <= 10 and rotb <= 10

filtered = [m for m in mols if passes_filter(m)]
print(f"{len(filtered)}/{len(mols)} pass basic drug-like filter")

# Print SMILES
for m in filtered[:20]:
    print(Chem.MolToSmiles(Chem.RemoveHs(m)))
```

Then prioritize candidates for:
1. Boltz-2 covalent docking (as in the paper — use PDB 3F75 for TgCPL)
2. QSAR model prediction (the AutoML ensemble from the paper)
3. Experimental IC50 testing

---

## Lead molecule reference

| Cpd | SMILES | TgCPL IC50 | HsCPL IC50 | Sel. | Notes |
|-----|--------|-----------|-----------|------|-------|
| 1 | `CCN(CC)CCN(Cc1ccc(F)cc1)c1nc(C#N)nc(N2CCOCC2)n1` | 320 nM | 2240 nM | 7x | Parent |
| 12 | `CCN(CC)CCN(c1nc(C#N)nc(N2CCOCC2)n1)C(c1ccc(F)cc1)C1CCCCC1` | 629 nM | 23245 nM | 37x | Cyclohexyl |
| 13 | `CCN(CC)CCN(c1nc(C#N)nc(N2CCOCC2)n1)C(c1ccc(F)cc1)C1CCCCCC1` | 497 nM | 19695 nM | 40x | Cycloheptyl |
| **18** | `N#Cc1nc(N2CCOCC2)nc(N(CCC2CCNCC2)C(c2ccc(F)cc2)C2CCCCC2)n1` | **210 nM** | 8080 nM | **38.5x** | **BEST LEAD** |

Warhead: reversible covalent nitrile → Cys31 (TgCPL) / Cys25 (HsCPL)  
Crystal structures: TgCPL = PDB 3F75 | HsCPL = PDB 5MAJ
