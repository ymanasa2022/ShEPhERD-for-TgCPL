# Running TgCPL ShEPhERD-2 on U-M Great Lakes

Account: `maom99`. Everything lives in scratch, not home — the conda env, the
PyTorch wheels and the HF checkpoints together are far too large for the home
quota.

```
/scratch/maom99_root/maom99/$USER/
├── shepherd2/              # cloned repo
├── envs/shepherd2/         # conda env (--prefix, not a named env)
├── hf_cache/               # HF_HOME: downloaded checkpoints
├── ShEPhERD-for-TgCPL/     # these scripts + tgcpl_generate.py
└── tgcpl_outputs/          # results
```

## 1. Upload

```bash
SCRATCH=/scratch/maom99_root/maom99/$USER
rsync -av ~/path/to/ShEPhERD-for-TgCPL/ uniqname@greatlakes.arc-ts.umich.edu:$SCRATCH/ShEPhERD-for-TgCPL/
```

## 2. Install (login node, once, ~15 min)

```bash
ssh uniqname@greatlakes.arc-ts.umich.edu
cd /scratch/maom99_root/maom99/$USER/ShEPhERD-for-TgCPL
bash greatlakes_setup.sh cu126      # or cu124 / cu128 — see note below
```

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

1. **Scratch path.** `ls /scratch/maom99_root/maom99/` — if your directory is
   not `$USER`, edit `SCRATCH` in `greatlakes_env.sh` and `greatlakes_setup.sh`.
2. **Anaconda module name.** `module avail anaconda`. The scripts try
   `python3.11-anaconda` then `anaconda`; if yours differs, edit the
   `module load` line in `greatlakes_env.sh` and `greatlakes_setup.sh`.
3. **CUDA version.** `nvidia-smi` on a GPU node shows the driver's CUDA. Pick
   the `cu*` extra at or below it. Great Lakes `gpu` partition has V100s;
   `spgpu` has newer A40s — `--partition=spgpu` may queue faster.
4. **Partition and limits.** Confirm `gpu` is right for your allocation, and
   that 8 h / 48 G / 8 CPUs are within your account's limits.

## Notes

- `conda activate` needs `source $(conda info --base)/etc/profile.d/conda.sh`
  first in a batch job — a non-interactive shell has no conda hook. This is the
  single most common reason a Slurm job dies instantly with
  `CommandNotFoundError`.
- `--batch-size 25` is a guess for V100 (16 GB) memory. If a task dies with
  CUDA OOM, drop to 10–15; it changes throughput only, not results.
- `HF_HOME` points at scratch so the checkpoint downloads once and is reused.
  The first job to run pays the download cost.
- Scratch is **purged** (typically 60 days untouched). Copy results you care
  about to Turbo or your OneDrive when a run finishes.
