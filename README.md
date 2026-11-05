# TgCPL ShEPhERD— Setup & Drug Design Guide

## Layout

```
scripts/
  tgcpl/                    importable package (RDKit only -- no torch)
    leads.py                reference compounds + registry-id lookup
    scaffolds.py            scaffold cores for --scaffold-constraint
    properties.py           descriptors, thresholds, rejection logic
    conformers.py           3D embedding (the one module needing shepherd)
    tgcpl_generate.py       generation driver
  generate/
    filter_generated.py     triage -> passing.smi + passing.sdf
    summarize_run.py        reporting over a run directory
    baseline_scores.py      null-model comparison
    compare_scaffold.sh     A/B two scaffold cores locally
setup/
  greatlakes_config.sh      ALL Great Lakes paths -- the one file to edit
  greatlakes_env.sh         activate; sourced by every job script
  greatlakes_setup.sh       one-shot install on Great Lakes
  local_env.sh              laptop/CPU counterpart
  setup_shepherd2.sh        portable installer for another cluster
slurm_scripts/
  greatlakes_redesign.sbatch   the 2x2 redesign campaign (array 1-4)
  greatlakes_tgcpl.sbatch      the original five runs (array 1-5)
  tgcpl.sbatch                 generic non-Great-Lakes fallback
tests/
  test_redesign.py          pre-flight: cores, sizing, filter. No GPU needed
  local_smoke_test.sh       end-to-end on CPU
  greatlakes_smoke_test.sh  end-to-end on a GPU node
TgCPL/
  lead_molecules.smi        SMILES + IC50/selectivity for all leads
```

The entry points add their own package directory to `sys.path`, so they run
from any working directory and no `PYTHONPATH` is needed.

Only `tgcpl_generate.py` and `conformers.py` need the shepherd stack.
Filtering, reporting and `tests/test_redesign.py` are RDKit-only, so results
can be triaged in a plain environment while generation runs on a cluster.

---

## Quick start (Great Lakes)

Paths live in `setup/greatlakes_config.sh`. Edit that one file; nothing else
hardcodes a path.

```bash
# 1. Check the configured paths resolve (instant)
bash -c 'source setup/greatlakes_config.sh && tgcpl_check_paths --pre-install'

# 2. Install: clones shepherd2, builds a python 3.11 + xtb env, pip installs
bash setup/greatlakes_setup.sh cu126        # ~15 min, login node, not sbatch

# 3. Pre-flight, no GPU
python tests/test_redesign.py               # expect ALL CHECKS PASSED

# 4. Smoke test on a GPU node
sbatch tests/greatlakes_smoke_test.sh

# 5. The campaign
RUN_STAMP=$(date +%Y%m%d_%H%M%S) \
    sbatch --export=ALL --array=1-4 slurm_scripts/greatlakes_redesign.sbatch
```

`#SBATCH --account=` in the three job scripts is the one value the config
cannot supply — Slurm parses those directives before the shell runs, so they
must be edited directly if your account changes.

On another cluster use `setup/setup_shepherd2.sh` and
`slurm_scripts/tgcpl.sbatch`, both of which hardcode their paths at the top.

## Quick start (laptop, CPU)

```bash
source setup/local_env.sh
bash tests/local_smoke_test.sh all
```

CPU only. Apple MPS does not work: `torch_cluster` and `torch_scatter` ship
CPU and CUDA kernels only, and `radius_graph` runs on every denoising step.

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
  → Poorly selective: the weakest of the series

Compound 12 / 13 (cyclohexyl / cycloheptyl benzylic):  
  → Bulky cycloalkyl LOCKS the benzylic carbon → forces 4-fluorobenzyl into S3  
  → P2 diethylamine now occupies S2 → interacts with TgCPL's Glu75 + Asp218  
  → These acidic residues are ABSENT in HsCPL → HsCPL loses affinity  
  → Mid-nanomolar TgCPL, and the best selectivity in the series ✓

Compound 18 (cyclohexyl P3 + piperidinylethyl P2):  
  → Same P3 mechanism PLUS piperidine NH engages Glu75/Asp218 more effectively  
  → The most potent lead, at comparable selectivity ✓✓ (BEST LEAD)
```

### Goals for ShEPhERD generation

#### Goal 1: Maintain or improve potency on TgCPL
- Keep the covalent nitrile warhead (Cys31 bond essential)
- P2 basic amine must reach Glu75 / Asp218 in S2
- Explore: longer P2 chains, different basic amines (pyrrolidine, piperazine, azepane), piperidine NH vs. NMe

#### Goal 2: Improve selectivity over HsCPL
- Maintain large cycloalkyl at benzylic P3 position (cyclohexyl/cycloheptyl optimal so far)
- Explore: bicyclic P3 groups (norbornyl, bicyclo[2.2.2]octyl) that fit TgCPL's narrower S3
- Avoid: gem-disubstitution at benzylic C (decreases both); polar heteroatoms at benzylic C (THP killed potency)
- R-configuration at benzylic carbon preferred over S

#### Goal 3: Bioisosteric exploration
- Replace 4-fluorobenzyl (P3 aryl) with other aryl/heteroaryl groups
- Replace morpholine (P1) with other saturated N-heterocycles
- Use `--neutralize-esp` to find bioisosteres across different charge states
- Use `--condition shape esp` (no pharmacophore) for more structurally diverse output

---

## ShEPhERD run strategy

### Run 1: Baseline — all 4 leads, full conditioning
```bash
python scripts/tgcpl/tgcpl_generate.py --ref-smiles all --n-samples 50 --condition all
```
Generates 200 molecules conditioned on shape + ESP + pharmacophores of each lead.
Best for finding close analogues with improved drug-like properties.

### Run 2: Shape + ESP from Cpd18 (most potent lead)
```bash
python scripts/tgcpl/tgcpl_generate.py --ref-smiles Cpd18 --n-samples 100 --condition shape esp
```
Removes pharmacophore constraint → more structural diversity while preserving
the 3D electrostatic profile matching Glu75/Asp218 interactions.

### Run 3: Grow from Cpd18 (+2 to +6 heavy atoms)
```bash
python scripts/tgcpl/tgcpl_generate.py --ref-smiles Cpd18 --n-samples 100 --add-atoms-range 2 6
```
Generates slightly larger molecules. Goal: explore extended P2 chains or
additional substituents that engage TgCPL-specific residues.

### Run 4: Scaffold-constrained from Cpd12
```bash
python scripts/tgcpl/tgcpl_generate.py --ref-smiles Cpd12 --n-samples 100 --scaffold-constraint
```
Fixes the triazine-CN-morpholine warhead atoms. Only varies the P2+P3 portions.
Guarantees the covalent warhead is preserved.

### Run 5: Dual-target selectivity — TgCPL AND NOT HsCPL ⭐
```bash
python scripts/tgcpl/tgcpl_generate.py --dual-target --dual-tgcpl-ref Cpd18 --dual-hscpl-ref Cpd1 \
    --n-samples 100 --scaffold-constraint
```
Uses ShEPhERD's `generate_composition()` with a signed composition:

```
weights_conditions = [+0.5, -0.5]
profiles = [profile_TgCPL_Cpd18, profile_HsCPL_Cpd1]
composition_mode = 'conditional'
```

This directly encodes **"generate molecules that look like Cpd18 in TgCPL AND do NOT look like Cpd1 in HsCPL"**. The −0.5 weight on the HsCPL profile repels generation away from the hydrophobic HsCPL S2 pocket features, while +0.5 on the TgCPL profile attracts toward the acidic TgCPL S2 pocket features.

**Why Cpd1 as the HsCPL reference?** Cpd1 is the series parent: potent on
HsCPL and the least selective lead, so its profile captures the undesirable
hydrophobic HsCPL S2 contacts the composition should steer away from.

> **Note on the assay numbers.** `scripts/tgcpl/leads.py` carries the
> experimental readouts, not the figures from the SAR write-up; the two differ
> for three of the four leads, most sharply for Cpd1. Cpd1 is still the least
> selective lead on either set, so this run's premise holds either way — but by
> a narrower margin than the write-up implies. Worth knowing before quoting a
> selectivity fold-change from this repo.

**Mechanistic interpretation**:
- TgCPL profile (Cpd18): encodes deep acidic S2 pocket; Glu75 + Asp218 contacts; narrow S2 geometry
- HsCPL profile (Cpd1): encodes shallow hydrophobic S2; Leu69 + Ala214 contacts; wider S2 geometry
- The AND NOT composition steers the diffusion denoising process toward TgCPL-fitting electrostatics and away from HsCPL-fitting hydrophobics

**Expected output**: molecules with novel scaffolds that maintain the triazine-nitrile warhead (scaffold constraint) but find new ways to project a basic amine toward TgCPL's acidic residues while avoiding bulky hydrophobic P2 groups that would fit HsCPL.

---

## Post-generation analysis

`filter_generated.py` does the triage. It globs recursively, so point it at a
run directory or a parent of several:

```bash
python scripts/generate/filter_generated.py "$OUTPUT_ROOT/redesign_<registry-id>_p1kept_<stamp>" \
    --max-mw 500 --max-rotb 10 --max-ring-size 7
```

It writes `passing.smi` and `passing.sdf` beside the input. The SDF keeps the
xTB-relaxed 3D coordinates, so it goes straight to docking without
re-embedding. It also prints a rejection tally, which is how you see *which*
constraint is binding rather than just the pass count.

**Warhead retention is checked by default** and is not optional: ShEPhERD
conditions on a 3D field, not a substructure, so nothing stops it discarding
the nitrile. Across the local runs to date the triazine survived in 2/10
unconstrained samples versus 11/11 scaffold-constrained ones.

Threshold notes, calibrated against the 446-compound records:

| flag | default | why |
|---|---|---|
| `--max-ring-size` | 7 | series histogram is `{5:1, 6:440, 7:5}` — 7 admits the real chemistry, rejects the macrocycles unconstrained generation produces |
| `--max-rotb` | 10 | the reference lead's own value; admits 98% of the series, whose median is 7. Tighten if it should actually bite |
| `--max-logp` | 5.0 | RDKit Crippen, which runs low on this chemotype versus commercial calculators — softer than the same number elsewhere |

Everything here is rule-based physicochemistry, **not ADMET prediction**. Real
absorption/clearance/hERG endpoints need a trained predictor. A pass means "not
obviously disqualified".

Then prioritize candidates for:
1. Boltz-2 covalent docking (as in the paper — use PDB 3F75 for TgCPL)
2. QSAR model prediction (the AutoML ensemble from the paper)
3. Experimental IC50 testing

For runs that free the P1 morpholine, add a covalent-reactivity check: the
nitrile's electrophilicity depends on how electron-poor the triazine is, and P1
tunes exactly that. A docking score will not catch a warhead that would not
react.

---

## Lead molecules

Structures and assay values are **not tracked by git**. `scripts/tgcpl/leads.py`
loads them at import from `TgCPL/leads.tsv`, which is gitignored.

A fresh clone has no such file and will raise on import. Give it one:

```bash
cp TgCPL/leads.example.tsv TgCPL/leads.tsv   # then fill in the real values
# or keep it outside the repo:
export TGCPL_LEADS=/path/to/leads.tsv
```

`TgCPL/leads.example.tsv` documents the columns and ships placeholder rows
(benzene, toluene) so the package imports and the tests run without any real
data present.

**This applies on the cluster too** — `git pull` will not bring the file, so
copy it across once:

```bash
scp TgCPL/leads.tsv greatlakes:$TGCPL_DIR/TgCPL/
```

Lead names used on the command line (`--ref-smiles Cpd12`) are the `name`
column; `python scripts/tgcpl/tgcpl_generate.py --help` lists what loaded.
Run directories are named by `vmcc`, so output joins back to the assay records.

Raw data lives outside this repo in `Data/records/` —
`experiment_readouts.csv` (assay values) and `metadata.csv` (computed
properties, registry ids).

Warhead: reversible covalent nitrile → Cys31 (TgCPL) / Cys25 (HsCPL)  
Crystal structures: TgCPL = PDB 3F75 | HsCPL = PDB 5MAJ
