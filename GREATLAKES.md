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
├── opt/shepherd2/                      # upstream coleygroup/shepherd2
└── campaigns/tgcpl-martin-campaign/scripts/ShEPhERD-for-TgCPL/   # this repo

# Scratch — large, regenerable, and PURGED after ~60 days untouched.
/scratch/maom_root/maom0/$USER/tgcpl-scratch-campaign/
├── SHEP_out/                           # OUTPUT_ROOT: generated molecules
├── intermediate_data/shep_hf_cache/    # HF_HOME: model checkpoints
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
bash greatlakes_setup.sh cu126      # or cu124 / cu128 — see note below
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
sbatch greatlakes_smoke_test.sh
```

Checks three things in order: torch sees the GPU and `xtb` is on PATH; the
repo's own benzene test runs end to end; a real TgCPL lead (Cpd18, 2 samples)
generates. **Do not skip this** — it catches a CUDA mismatch in 10 minutes
instead of 8 hours into a full run.

## 4. Production runs

```bash
sbatch greatlakes_tgcpl.sh          # all five
sbatch --array=3 greatlakes_tgcpl.sh  # just run 3
```

| Task | Reference | Conditioning | Samples |
|------|-----------|--------------|---------|
| 1 | all 8 leads | shape+ESP+pharm | 50 each (400) |
| 2 | Cpd18 | shape+ESP only | 100 |
| 3 | Cpd18 | all, grown 2–6 atoms | 100 |
| 4 | Cpd12 | all, scaffold-fixed | 100 |
| 5 | Cpd18 vs Cpd1 | dual-target, scaffold-fixed | 100 |

A job array rather than one script running five jobs back to back: they run in
parallel, each gets the full 8 h, and one failure doesn't take the rest down.

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
   `greatlakes_redesign.sh`, `greatlakes_tgcpl.sh` and
   `greatlakes_smoke_test.sh` must match it exactly or sbatch refuses the job.
   These are `#SBATCH` directives, parsed before the shell runs, so they
   **cannot** read `$ACCOUNT` from `greatlakes_config.sh` — they are the one
   place a value is duplicated. `maom0` is inferred from the scratch path
   (`/scratch/maom_root/maom0/...`); verify it.
2. **Paths.** `bash -c 'source greatlakes_config.sh && tgcpl_check_paths'`
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
