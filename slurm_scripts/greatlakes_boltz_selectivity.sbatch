#!/usr/bin/env bash
# Dual-target Boltz-2 co-folding: folds each ligand against both enzymes and
# subtracts the affinities, so the readout is a selectivity window.
# Inputs come from scripts/dock/boltz_prep.py; aggregate afterwards with
# scripts/dock/boltz_selectivity.py (or re-run this file with --aggregate).
#
#   task 1  TgCPL (3F75)
#   task 2  HsCPL (5MAJ)
#
# ONE-TIME SETUP (boltz is NOT in the shepherd env -- it pins its own torch and
# installing it there breaks torch_cluster):
#   source setup/greatlakes_config.sh
#   conda create -y --prefix "$BOLTZ_ENV" -c conda-forge python=3.11
#   conda activate "$BOLTZ_ENV" && pip install boltz -U
#   bash scripts/dock/fetch_targets.sh        # if data/targets/ is missing
#
# COST: one co-folding per ligand per target, so check `wc -l passing.smi`
# before prepping and use boltz_prep.py --limit if that number is large.
#SBATCH --job-name=tgcpl_boltz
#SBATCH --account=maom0          # must match your Slurm account: run `my_accounts`
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=1-2
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Paths only -- do NOT source greatlakes_env.sh: it activates the shepherd env,
# and this job wants the boltz env.
source "$REPO_ROOT/setup/greatlakes_config.sh"
source "$CONDA_SH"

if [ ! -d "$BOLTZ_ENV" ]; then
    echo "ERROR: no boltz env at $BOLTZ_ENV" >&2
    echo "  See the ONE-TIME SETUP block at the top of this file." >&2
    exit 1
fi
conda activate "$BOLTZ_ENV"

if ! command -v boltz >/dev/null; then
    echo "ERROR: boltz not on PATH after activating $BOLTZ_ENV" >&2
    exit 1
fi

IN_ROOT="$BOLTZ_ROOT/boltz_in"
OUT_ROOT="$BOLTZ_ROOT/boltz_out"
mkdir -p "$OUT_ROOT" "$BOLTZ_CACHE"

# `... .sbatch --aggregate` for the dependent follow-up job. Needs RDKit, not
# boltz, so it borrows the shepherd env.
if [ "${1:-}" = "--aggregate" ]; then
    conda deactivate
    conda activate "$ENV_PREFIX"
    python "$TGCPL_DIR/scripts/dock/boltz_selectivity.py" "$OUT_ROOT" \
        --ligands "$IN_ROOT/ligands.tsv"
    exit 0
fi

case "${SLURM_ARRAY_TASK_ID:?set --array, e.g. sbatch --array=1-2}" in
  1) TARGET=tgcpl; PDB=3F75 ;;
  2) TARGET=hscpl; PDB=5MAJ ;;
  *) echo "ERROR: unknown array index ${SLURM_ARRAY_TASK_ID} (expected 1-2)" >&2
     exit 1 ;;
esac

IN_DIR="$IN_ROOT/$TARGET"
OUT_DIR="$OUT_ROOT/$TARGET"

if [ ! -d "$IN_DIR" ]; then
    echo "ERROR: no inputs at $IN_DIR" >&2
    echo "  Run scripts/dock/boltz_prep.py first -- see the header." >&2
    exit 1
fi

N_YAML=$(find "$IN_DIR" -name '*.yaml' | wc -l | tr -d ' ')

echo "=== task ${SLURM_ARRAY_TASK_ID} (${TARGET}, PDB ${PDB}) on $HOSTNAME ==="
echo "started : $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "inputs  : $N_YAML YAML in $IN_DIR"
echo "output  : $OUT_DIR"
echo "cache   : $BOLTZ_CACHE"
echo

# One invocation per directory, not per ligand: every YAML here carries the same
# protein sequence, so the MSA is fetched once instead of N times.
# --use_msa_server needs outbound network from the compute node; if blocked,
# generate the a3m elsewhere and re-run boltz_prep.py --msa-dir.
boltz predict "$IN_DIR" \
    --out_dir "$OUT_DIR" \
    --use_msa_server \
    --cache "$BOLTZ_CACHE" \
    --diffusion_samples 3 \
    --recycling_steps 3 \
    --accelerator gpu

echo
echo "=== task ${SLURM_ARRAY_TASK_ID} (${TARGET}) finished: $(date) ==="
echo
echo "When BOTH tasks are done:"
echo "  python $TGCPL_DIR/scripts/dock/boltz_selectivity.py $OUT_ROOT \\"
echo "      --ligands $IN_ROOT/ligands.tsv"
echo
echo "Read the CONTROLS block it prints before the ranked list. If the four"
echo "leads do not come out in roughly their measured selectivity order, the"
echo "ranking underneath them is not worth acting on."
