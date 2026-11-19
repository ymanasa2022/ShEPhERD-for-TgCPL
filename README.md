# TgCPL ShEPhERD— Setup & Drug Design Guide

## Layout

```
scripts/
  tgcpl/                    importable package (RDKit only -- no torch)
    leads.py                reference compounds + registry-id lookup
    scaffolds.py            scaffold cores for --scaffold-constraint
    properties.py           descriptors, thresholds, rejection logic
    selectivity.py          SAR-derived TgCPL-over-HsCPL substructure rules
    conformers.py           3D embedding (the one module needing shepherd)
    tgcpl_generate.py       generation driver
  generate/
    filter_generated.py     triage -> passing.smi + passing.sdf
    selectivity_score.py    rank by selectivity -> selectivity.tsv + selected.sdf
    summarize_run.py        reporting over a run directory
    baseline_scores.py      null-model comparison
    compare_scaffold.sh     A/B two scaffold cores locally
  dock/
    fetch_targets.sh        pull 3F75 / 5MAJ sequences from RCSB
    boltz_prep.py           selected.sdf -> Boltz-2 YAML for BOTH targets
    boltz_selectivity.py    pair the two runs, rank by predicted window
setup/
  greatlakes_config.sh      ALL Great Lakes paths -- the one file to edit
  greatlakes_env.sh         activate; sourced by every job script
  greatlakes_setup.sh       one-shot install on Great Lakes
  local_env.sh              laptop/CPU counterpart
  setup_shepherd2.sh        portable installer for another cluster
slurm_scripts/
  greatlakes_redesign.sbatch   the 2x2 redesign campaign (array 1-4)
  greatlakes_boltz_selectivity.sbatch  dual-target docking (array 1-2)
  greatlakes_tgcpl.sbatch      the original five runs (array 1-5)
  tgcpl.sbatch                 generic non-Great-Lakes fallback
tests/
  test_redesign.py          pre-flight: cores, sizing, filter, selectivity
  local_smoke_test.sh       end-to-end on CPU
  greatlakes_smoke_test.sh  end-to-end on a GPU node
data/
  lead_molecules.smi        SMILES + IC50/selectivity for all leads
  targets/                  mature TgCPL / HsCPL sequences for docking
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
1. The selectivity study below (ligand-based triage, then dual-target docking)
2. QSAR model prediction (the AutoML ensemble from the paper)
3. Experimental IC50 testing

---

## Selectivity study (TgCPL vs HsCPL)

Selectivity is decided at the **docking** stage, not during generation.
Generation conditions on Cpd12 and Cpd18 — both TgCPL-selective leads — and
everything that survives triage gets folded against both enzymes.

```
greatlakes_redesign.sbatch --array=2,4   ->  passing.sdf
                                                  |
                                             boltz_prep.py
                                                  |
                                      boltz predict (TgCPL) + (HsCPL)
                                                  |
                                         boltz_selectivity.py
```

### Generate

Tasks 2 and 4 of the existing redesign campaign are Cpd12 and Cpd18 with the
P1 morpholine pinned:

```bash
RUN_STAMP=$(date +%Y%m%d_%H%M%S) \
    sbatch --export=ALL --array=2,4 slurm_scripts/greatlakes_redesign.sbatch
```

Use `--array=1-4` instead to also get the two arms that free P1. Those explore
S1 but change the nitrile's electrophilicity, so they need a covalent-reactivity
check on top of a docking score.

Each task ends by running `filter_generated.py`, which writes `passing.sdf`
with xTB-relaxed coordinates — that file is the docking input directly.

### Dock against both targets

```bash
bash scripts/dock/fetch_targets.sh                       # once
RUN=$OUTPUT_ROOT/redesign_VMCC-0005222_p1kept_<stamp>    # Cpd18's run
python scripts/dock/boltz_prep.py "$RUN/passing.sdf" -o "$BOLTZ_ROOT/boltz_in"
# sbatch --array=1-2 slurm_scripts/greatlakes_boltz_selectivity.sbatch
python scripts/dock/boltz_selectivity.py "$BOLTZ_ROOT/boltz_out" \
    --ligands "$BOLTZ_ROOT/boltz_in/ligands.tsv"
```

Check `wc -l "$RUN/passing.smi"` first. Every ligand costs two co-foldings, so
pass `--limit N` to `boltz_prep.py` if that number is large.

`affinity_pred_value` is log10(IC50 / µM), so the readout is

```
d_logIC50 = hscpl_value - tgcpl_value        positive = TgCPL-preferring
fold      = 10 ** d_logIC50
```

**The four leads are folded in as controls by default**, under `CTRL_<name>`.
They have measured IC50s against both enzymes, so `boltz_selectivity.py` prints
the predicted-vs-measured comparison *above* the ranked list. Read it first.
Boltz-2's affinity head was trained to rank actives against a single target;
differencing two independent predictions across two homologues is off-label,
and the two heads are not guaranteed to share a zero. Four congeneric controls
cannot validate that — but they can invalidate it. If the control correlation
comes out flat or inverted, the right result to report is "this did not
transfer", not a top-10 list.

Also true of every number here: nothing models the covalent bond to
Cys31/Cys25. These are non-covalent poses of a covalent inhibitor, so they
speak to pocket complementarity and not to the real binding mode.

### Optional: ligand-based pre-ranking

`selectivity_score.py` scores molecules for selectivity *without* a protein,
using substructure rules from the series SAR (`tgcpl/selectivity.py`) and
3D ShEPhERD-score similarity to the selective leads minus the non-selective
one. It is not part of the pipeline above — docking is the readout — but it
is a cheap way to choose which molecules to dock when `passing.sdf` is larger
than the GPU budget:

```bash
python scripts/generate/selectivity_score.py "$RUN" --3d --require-series --top 40
python scripts/dock/boltz_prep.py "$RUN/selected.sdf" -o "$BOLTZ_ROOT/boltz_in"
```

Treat its output as a triage order, not a prediction. The rules are hand-set
from four compounds and reward resemblance to the one chemotype already known
to work, so they penalise exactly the novelty generation exists to find. And
measured on the leads, whole-molecule ESP separates this congeneric series by
only ~0.07 — which is why selectivity is decided by docking here and not by
ligand similarity.

For runs that free the P1 morpholine, add a covalent-reactivity check: the
nitrile's electrophilicity depends on how electron-poor the triazine is, and P1
tunes exactly that. A docking score will not catch a warhead that would not
react.

---

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
