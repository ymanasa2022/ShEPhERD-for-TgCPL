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
data/
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

---

## Generating from Cpd12 and Cpd18

Tasks 2 and 4 of the redesign campaign are the two TgCPL-selective leads with
the P1 morpholine pinned:

```bash
RUN_STAMP=$(date +%Y%m%d_%H%M%S) \
    sbatch --export=ALL --array=2,4 slurm_scripts/greatlakes_redesign.sbatch
```

| task | reference | core |
|---|---|---|
| 2 | Cpd12 | `warhead_morpholine_p2n` |
| 4 | Cpd18 | `warhead_morpholine_p2n` |

Both pin the P2 nitrogen, so its two substituents regrow in every sample while
the warhead and P1 stay fixed. Use `--array=1-4` to also get the arms that free
P1: those explore S1 but change the nitrile's electrophilicity, so they need a
covalent-reactivity check rather than a docking score alone.

Each task ends by running `filter_generated.py`, which writes `passing.smi` and
`passing.sdf` (xTB-relaxed coordinates, ready for docking).

## Lead molecules

`scripts/tgcpl/leads.py` loads them at import from `data/leads.tsv`, which is
tracked, so a clone or `git pull` brings it. `TGCPL_LEADS` overrides the path
if you want to run against a different set.

`data/leads.example.tsv` documents the columns and carries placeholder rows.

Lead names used on the command line (`--ref-smiles Cpd12`) are the `name`
column; `python scripts/tgcpl/tgcpl_generate.py --help` lists what loaded.
Run directories are named by `vmcc`, so output joins back to the assay records.

Raw data lives outside this repo in `Data/records/` —
`experiment_readouts.csv` (assay values) and `metadata.csv` (computed
properties, registry ids).

Warhead: reversible covalent nitrile → Cys31 (TgCPL) / Cys25 (HsCPL)  
Crystal structures: TgCPL = PDB 3F75 | HsCPL = PDB 5MAJ
