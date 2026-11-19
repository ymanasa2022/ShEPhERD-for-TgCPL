# Running TgCPL ShEPhERD-2 on U-M Great Lakes

Account: `maom`. **All paths live in `greatlakes_config.sh`** — that is the one
file to edit, and both the setup script and every job script source it. Nothing
else hardcodes a path.

Work is split across two filesystems:

```
# Turbo — persistent. Anything expensive to rebuild.
/home/ymanasa/turbo/ymanasa/
├── miniconda3/                         # conda install (CONDA_SH)
├── micromamba/envs/shepherd2/          # ENV_PREFIX: the env itself
├── micromamba/envs/boltz2/             # BOLTZ_ENV: the docking stage
├── opt/shepherd2/                      # upstream coleygroup/shepherd2
└── campaigns/tgcpl-martin-campaign/scripts/ShEPhERD-for-TgCPL/   # this repo

# Scratch — large, regenerable, and PURGED after ~60 days untouched.
/scratch/maom_root/maom0/$USER/tgcpl-scratch-campaign/
├── SHEP_out/                           # OUTPUT_ROOT: generated molecules
├── boltz/                              # BOLTZ_ROOT: boltz_in/, boltz_out/
├── intermediate_data/shep_hf_cache/    # HF_HOME: model checkpoints
├── intermediate_data/boltz_cache/      # BOLTZ_CACHE: boltz weights
└── logs/
```

The env lives on Turbo on purpose: it is thousands of files that nothing
"touches" in the sense the purge policy means, so on scratch it would quietly
vanish and cost a ~15 min torch/CUDA reinstall. Only regenerable output sits on
scratch.

## 1. Get the code there

```bash
git clone https://github.com/ymanasa2022/ShEPhERD-for-TgCPL.git \
    /home/ymanasa/turbo/ymanasa/campaigns/tgcpl-martin-campaign/scripts/ShEPhERD-for-TgCPL
# already cloned? just:  git pull
```

## 2. Install (login node, once, ~15 min)

```bash
ssh greatlakes
cd /home/ymanasa/turbo/ymanasa/campaigns/tgcpl-martin-campaign/scripts/ShEPhERD-for-TgCPL
bash setup/greatlakes_setup.sh cu126   # or cu124 / cu128 — see note below
```

Setup validates every configured path first and refuses to continue if one is
missing, rather than failing partway through. It also verifies that
`conda activate` actually took before installing — otherwise pip silently falls
back to a user install and scatters torch through your home quota.

Two things that will bite if changed: the env must be **Python 3.11**
(`shepherd` requires `>=3.11,<3.12`; 3.12 fails at metadata resolution), and
`xtb` must come from conda-forge — generation dies partway with
`FileNotFoundError: 'xtb'` without it.

## 3. Smoke test (one 30-min GPU job)

```bash
sbatch tests/greatlakes_smoke_test.sh
```

Checks three things in order: torch sees the GPU and `xtb` is on PATH; the
repo's own benzene test runs end to end; a real TgCPL lead (Cpd18, 2 samples)
generates. **Do not skip this** — it catches a CUDA mismatch in 10 minutes
instead of 8 hours into a full run.

## 4. Production runs

```bash
sbatch slurm_scripts/greatlakes_tgcpl.sbatch            # all five
sbatch --array=3 slurm_scripts/greatlakes_tgcpl.sbatch  # just run 3
```

| Task | Reference | Conditioning | Samples |
|------|-----------|--------------|---------|
| 1 | all 4 leads | shape+ESP+pharm | 50 each (200) |
| 2 | Cpd18 | shape+ESP only | 100 |
| 3 | Cpd18 | all, grown 2–6 atoms | 100 |
| 4 | Cpd12 | all, scaffold-fixed | 100 |
| 5 | Cpd18 vs Cpd1 | dual-target, scaffold-fixed | 100 |

A job array rather than one script running five jobs back to back: they run in
parallel, each gets the full 8 h, and one failure doesn't take the rest down.

## 5. The selectivity study

Generation is plain TgCPL conditioning on the two selective leads -- tasks 2
and 4 of the redesign campaign, Cpd12 and Cpd18 with P1 pinned. Selectivity is
decided downstream, by docking against both enzymes.

```bash
RUN_STAMP=$(date +%Y%m%d_%H%M%S) \
    sbatch --export=ALL --array=2,4 slurm_scripts/greatlakes_redesign.sbatch
```

Then the docking stage. **One-time install**, in its own env -- boltz pins its
own torch, and installing it into the shepherd env breaks
`torch_cluster`/`torch_scatter`:

```bash
source setup/greatlakes_config.sh
source "$CONDA_SH"
conda create -y --prefix "$BOLTZ_ENV" -c conda-forge python=3.11
conda activate "$BOLTZ_ENV" && pip install boltz -U
bash scripts/dock/fetch_targets.sh          # needs outbound network
```

Per run, folding every passing molecule against both targets:

```bash
RUN=$OUTPUT_ROOT/redesign_VMCC-0005222_p1kept_<stamp>     # Cpd18's run
wc -l "$RUN/passing.smi"                                  # check the count first
python scripts/dock/boltz_prep.py "$RUN/passing.sdf" -o "$BOLTZ_ROOT/boltz_in"

jid=$(sbatch --parsable --array=1-2 slurm_scripts/greatlakes_boltz_selectivity.sbatch)
sbatch --dependency=afterok:$jid slurm_scripts/greatlakes_boltz_selectivity.sbatch --aggregate
```

Two things to check before scaling up:

- **Outbound network from the GPU node.** `--use_msa_server` reaches
  `api.colabfold.com`. One `boltz predict` call per target directory fetches
  the MSA once for all its ligands, since they share a protein sequence -- but
  if the node is firewalled, generate the a3m elsewhere and re-run
  `boltz_prep.py --msa-dir`.
- **Cost.** One co-folding per ligand per target, plus 4 controls. Use
  `boltz_prep.py --limit N` until the controls look sane.

## Monitoring

```bash
squeue -u $USER                     # queued/running
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ReqTRES%40
tail -f tgcpl_gen_<jobid>_<task>.out
scancel <jobid>
```

## Things to verify on first login

These are the parts I could not check from your laptop:

1. **Slurm account.** Run `my_accounts`. The `#SBATCH --account=` line in
   `slurm_scripts/greatlakes_redesign.sbatch`,
   `slurm_scripts/greatlakes_tgcpl.sbatch`,
   `slurm_scripts/greatlakes_boltz_selectivity.sbatch` and
   `tests/greatlakes_smoke_test.sh` must match it exactly or sbatch refuses the job.
   These are `#SBATCH` directives, parsed before the shell runs, so they
   **cannot** read `$ACCOUNT` from `greatlakes_config.sh` — they are the one
   place a value is duplicated. `maom0` is inferred from the scratch path
   (`/scratch/maom_root/maom0/...`); verify it.
2. **Paths.** `bash -c 'source setup/greatlakes_config.sh && tgcpl_check_paths'`
   prints anything missing. Everything else reads from that one file.
3. **CUDA version.** `nvidia-smi` on a GPU node shows the driver's CUDA. Pick
   the `cu*` extra at or below it. Great Lakes `gpu` partition has V100s;
   `spgpu` has newer A40s — `--partition=spgpu` may queue faster.
4. **Partition and limits.** Confirm `gpu` is right for your allocation, and
   that 8 h / 48 G / 8 CPUs are within your account's limits.

## Notes

- `conda activate` needs a hook sourced first in a batch job — a
  non-interactive shell has no conda function. `greatlakes_env.sh` sources
  `$CONDA_SH` explicitly rather than deriving it from `conda info --base`,
  because the inherited `CONDA_EXE` points at `/mnt/turbo/...`, a mount path
  that no longer exists (Turbo is at `/nfs/turbo`). Deriving the hook picks up
  whichever conda is on PATH, which is not necessarily the one holding
  `ENV_PREFIX`. Symptom of getting this wrong, interactively or in a job:
  `bash: /mnt/turbo/.../conda: No such file or directory`.
- The env must be **Python 3.11**. `shepherd` declares `>=3.11,<3.12`, so
  installing from a 3.12 base fails with
  `Package 'shepherd' requires a different Python`.
- `--batch-size 25` is a guess for V100 (16 GB) memory. If a task dies with
  CUDA OOM, drop to 10–15; it changes throughput only, not results.
- `HF_HOME` points at scratch so the checkpoint downloads once and is reused.
  The first job to run pays the download cost.
- Scratch is **purged** (typically 60 days untouched). Copy results you care
  about to Turbo or your OneDrive when a run finishes.
